"""
OpenRouter LLM Service Handler
Handles all interactions with OpenRouter API for RAG-based document QA.
Uses the OpenAI-compatible API via OpenRouter's endpoint.

Supports easy model switching via OPENROUTER_MODEL env variable.
"""

import logging
import os
from typing import List, Dict, Any, Optional
from openai import OpenAI
import asyncio
from functools import wraps

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────
# Popular OpenRouter models — change OPENROUTER_MODEL in .env
# to any of these (or any model listed on openrouter.ai/models)
# ──────────────────────────────────────────────────────────────
AVAILABLE_MODELS = {
    # Free-tier models (no credits needed)
    "qwen/qwen3.6-plus:free": "Qwen 3.6 Plus (Free)",
    "nvidia/nemotron-3-super:free": "Nemotron 3 Super (Free)",
    "nvidia/nemotron-3-nano-30b-a3b:free": "Nemotron 3 Nano 30B (Free)",
    "stepfun/step-3.5-flash:free": "Step 3.5 Flash (Free)",
    "z-ai/glm-4.5-air:free": "GLM 4.5 Air (Free)",
    "arcee-ai/trinity-large-preview:free": "Trinity Large Preview (Free)",
    "arcee-ai/trinity-mini:free": "Trinity Mini (Free)",

    # Auto-router (picks best free model for your request)
    "openrouter/free": "Auto Free Router",

    # Paid but affordable
    "google/gemini-2.0-flash-001": "Gemini 2.0 Flash",
    "deepseek/deepseek-chat-v3-0324": "DeepSeek V3",
    "mistralai/mistral-small-3.1-24b-instruct": "Mistral Small 3.1 24B",
    "anthropic/claude-3.5-haiku": "Claude 3.5 Haiku",

    # Premium
    "anthropic/claude-3.5-sonnet": "Claude 3.5 Sonnet",
    "openai/gpt-4o": "GPT-4o",
    "google/gemini-2.5-pro-preview-03-25": "Gemini 2.5 Pro",
}

# Default model if none specified — uses OpenRouter's free auto-router
DEFAULT_MODEL = "openrouter/free"


def async_wrap(func):
    """Decorator to make sync functions async"""

    @wraps(func)
    async def run(*args, **kwargs):
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, lambda: func(*args, **kwargs))

    return run


class OpenRouterService:
    """Service class for OpenRouter API interactions (OpenAI-compatible)"""

    def __init__(
            self,
            api_key: Optional[str] = None,
            model_name: Optional[str] = None,
            app_name: Optional[str] = None,
            app_url: Optional[str] = None,
    ):
        """
        Initialize OpenRouter Service

        Args:
            api_key: OpenRouter API key (starts with sk-or-)
            model_name: Model identifier (e.g. 'meta-llama/llama-3.1-8b-instruct:free')
            app_name: Your application name (shown on OpenRouter dashboard)
            app_url: Your application URL (for OpenRouter rankings)
        """
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        self.model_name = model_name or os.getenv("OPENROUTER_MODEL", DEFAULT_MODEL)
        self.app_name = app_name or os.getenv("OPENROUTER_APP_NAME", "Microsoft RAG QA")
        self.app_url = app_url or os.getenv("OPENROUTER_APP_URL", "")
        self.base_url = "https://openrouter.ai/api/v1"

        # Validate required credentials
        if not self.api_key:
            raise ValueError(
                "OPENROUTER_API_KEY environment variable is required. "
                "Get your key at https://openrouter.ai/keys"
            )

        # Initialize client
        self.client = None
        self._initialize_client()

    def _initialize_client(self):
        """Initialize OpenRouter client using the OpenAI SDK"""
        try:
            headers = {}
            if self.app_url:
                headers["HTTP-Referer"] = self.app_url
            if self.app_name:
                headers["X-Title"] = self.app_name

            self.client = OpenAI(
                base_url=self.base_url,
                api_key=self.api_key,
                default_headers=headers,
            )

            # Test connection with a simple request
            try:
                test_response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": "Test"}],
                    max_tokens=10,
                    temperature=0.1,
                )
                if test_response:
                    logger.info(f"✅ Successfully initialized OpenRouter with model: {self.model_name}")
                else:
                    logger.warning(f"⚠️ OpenRouter returned empty test response, but client is ready")
            except Exception as test_err:
                logger.warning(f"⚠️ OpenRouter test call failed ({test_err}), but client is initialized. Will retry on first real request.")

        except Exception as e:
            logger.error(f"❌ Failed to initialize OpenRouter client: {e}")
            self.client = None
            raise

    # ── public helpers ──────────────────────────────────────────

    def switch_model(self, new_model: str) -> bool:
        """
        Hot-switch the active model without restarting the server.

        Args:
            new_model: OpenRouter model identifier

        Returns:
            True on success
        """
        old_model = self.model_name
        self.model_name = new_model
        try:
            # Quick validation call
            self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": "Test"}],
                max_tokens=5,
                temperature=0.1,
            )
            logger.info(f"🔄 Model switched: {old_model} → {new_model}")
            return True
        except Exception as e:
            logger.error(f"❌ Model switch failed, reverting: {e}")
            self.model_name = old_model
            return False

    @staticmethod
    def list_available_models() -> Dict[str, str]:
        """Return the curated list of popular models"""
        return AVAILABLE_MODELS

    # ── RAG answer generation ──────────────────────────────────

    async def generate_rag_answer(
            self,
            question: str,
            relevant_chunks: List[Dict[str, Any]],
            document_id: str,
            temperature: float = 0.3,
            max_tokens: int = 800,
    ) -> Dict[str, Any]:
        """
        Generate answer using RAG approach with OpenRouter

        Args:
            question: User's question
            relevant_chunks: List of relevant document chunks
            document_id: Document identifier
            temperature: Sampling temperature (0-1)
            max_tokens: Maximum tokens in response

        Returns:
            Dict containing answer, confidence, sources, and metadata
        """
        if not self.client:
            return {
                "answer": "OpenRouter service not available",
                "confidence": 0.0,
                "sources": [],
                "chunks_retrieved": 0,
            }

        try:
            logger.info(f"🔍 Retrieved {len(relevant_chunks)} chunks for question: {question[:50]}...")

            if not relevant_chunks:
                logger.warning(f"⚠️ No context retrieved for document: {document_id}")
                return {
                    "answer": "No relevant context found in the document to answer this question.",
                    "confidence": 0.0,
                    "sources": [],
                    "chunks_retrieved": 0,
                }

            # Log top retrieved chunks
            for i, chunk in enumerate(relevant_chunks[:3]):
                logger.info(
                    f"📄 Chunk {i + 1} (score: {chunk.get('similarity_score', 0):.3f}): {chunk['text'][:100]}..."
                )

            # Build context from chunks
            context = self._build_context(relevant_chunks)

            # Create RAG prompt
            system_prompt, user_prompt = self._create_rag_prompts(question, context)

            # Generate response via OpenRouter
            response = await self._generate_completion(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=temperature,
                max_tokens=max_tokens,
            )

            if response:
                answer_text = response.strip()
                confidence = self._estimate_confidence(relevant_chunks)

                # Extract sources
                sources = [
                    {
                        "chunk_id": chunk["chunk_id"],
                        "similarity_score": chunk["similarity_score"],
                        "preview": chunk["text"][:200] + "..." if len(chunk["text"]) > 200 else chunk["text"],
                    }
                    for chunk in relevant_chunks[:3]
                ]

                logger.info(f"✅ Generated answer with confidence {confidence:.2f}")

                return {
                    "answer": answer_text,
                    "confidence": confidence,
                    "sources": sources,
                    "chunks_retrieved": len(relevant_chunks),
                    "model_used": self.model_name,
                }
            else:
                logger.error("❌ OpenRouter returned empty response")
                return {
                    "answer": "Unable to generate response - empty response from OpenRouter",
                    "confidence": 0.0,
                    "sources": [],
                    "chunks_retrieved": len(relevant_chunks),
                }

        except Exception as e:
            logger.error(f"❌ Error generating RAG answer: {e}")
            return {
                "answer": f"Error generating answer: {str(e)}",
                "confidence": 0.0,
                "sources": [],
                "chunks_retrieved": len(relevant_chunks),
            }

    # ── private helpers ─────────────────────────────────────────

    async def _generate_completion(
            self,
            system_prompt: str,
            user_prompt: str,
            temperature: float = 0.2,
            max_tokens: int = 500,
    ) -> Optional[str]:
        """
        Generate completion using OpenRouter Chat API (OpenAI-compatible)
        """
        try:
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=temperature,
                    max_tokens=max_tokens,
                    top_p=0.8,
                    frequency_penalty=0.0,
                    presence_penalty=0.0,
                ),
            )

            if response and response.choices:
                return response.choices[0].message.content
            return None

        except Exception as e:
            logger.error(f"Error calling OpenRouter API: {e}")
            raise

    def _build_context(self, chunks: List[Dict[str, Any]], max_length: int = 10000) -> str:
        """Build context string from retrieved chunks"""
        context_parts = []
        total_length = 0

        for i, chunk in enumerate(chunks):
            chunk_text = f"[Context {i + 1}]\n{chunk['text']}\n"

            if total_length + len(chunk_text) > max_length:
                break

            context_parts.append(chunk_text)
            total_length += len(chunk_text)

        return "\n".join(context_parts)

    def _create_rag_prompts(self, question: str, context: str) -> tuple[str, str]:
        """Create system and user prompts for RAG"""
        system_prompt = """You are a helpful AI assistant who explains documents in a friendly, conversational way.

    YOUR PERSONALITY:
    - Talk naturally like a knowledgeable friend helping someone understand something
    - Be warm, approachable, and patient
    - Explain things clearly so anyone can understand, regardless of their background
    - Use everyday language - avoid jargon unless necessary (then explain it)

    RESPONSE FORMAT:
    - Use Markdown to make your response easy to read and visually appealing
    - Structure your answer logically with clear sections
    - Use appropriate formatting:
      * **Bold text** for key concepts or important terms
      * `code formatting` for technical terms, file names, or specific values
      * Bullet points for lists or multiple items
      * Numbered lists for sequential steps or procedures
      * > Blockquotes for tips, warnings, or important notes
      * ### Subheadings to organize different aspects of your answer

    HOW TO ANSWER:
    - Start with a direct, clear answer to the question
    - Then provide context and explanation to help them understand WHY
    - If there are multiple aspects, break them down into digestible parts
    - Use examples or analogies when helpful
    - Connect information in a way that builds understanding
    - End with a summary or key takeaway if the answer is complex

    TONE GUIDELINES:
    - Natural and conversational (like talking to a colleague over coffee)
    - Professional but not stiff or overly formal
    - Helpful and educational
    - Confident but humble - if something isn't in the document, say so clearly

    WHAT TO AVOID:
    - Don't sound like you're writing a textbook or essay
    - Don't use phrases like "according to the document" or "the context states"
    - Don't overwhelm with too much information at once
    - Don't make up information - only use what's in the provided context

    If the document doesn't contain the answer, say something like:
    "I don't see information about that in this document. The document focuses on [what it does cover], but doesn't mention [what they asked about]."
    """

        user_prompt = f"""Here's the relevant information from the document:

    {context}

    ---

    The user wants to know: **{question}**

    Please provide a helpful, well-formatted explanation that makes this easy to understand:"""

        return system_prompt, user_prompt

    def _estimate_confidence(self, chunks: List[Dict[str, Any]]) -> float:
        """Estimate confidence based on similarity scores"""
        if not chunks:
            return 0.0

        weights = [1.0, 0.8, 0.6, 0.4, 0.2]
        total_score = 0.0
        total_weight = 0.0

        for i, chunk in enumerate(chunks[:5]):
            weight = weights[i] if i < len(weights) else 0.1
            total_score += chunk["similarity_score"] * weight
            total_weight += weight

        confidence = total_score / total_weight if total_weight > 0 else 0.0
        return min(confidence, 1.0)

    # ── connection test & info ──────────────────────────────────

    async def test_connection(self) -> Dict[str, Any]:
        """Test OpenRouter connection and functionality"""
        try:
            if not self.client:
                return {
                    "status": "error",
                    "message": "OpenRouter client not initialized",
                }

            test_context = """
Context: Artificial Intelligence (AI) is a broad field of computer science 
that aims to create machines capable of performing tasks that typically 
require human intelligence.
"""
            test_question = "What is AI?"

            system_prompt, user_prompt = self._create_rag_prompts(test_question, test_context)

            response = await self._generate_completion(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=0.1,
                max_tokens=100,
            )

            return {
                "status": "success",
                "message": "OpenRouter API is working with RAG",
                "model_name": self.model_name,
                "model_display_name": AVAILABLE_MODELS.get(self.model_name, self.model_name),
                "test_response": response if response else "No response text",
                "base_url": self.base_url,
            }

        except Exception as e:
            return {
                "status": "error",
                "message": f"OpenRouter test failed: {str(e)}",
            }

    def get_service_info(self) -> Dict[str, Any]:
        """Get service configuration information"""
        return {
            "service": "OpenRouter",
            "model_available": self.client is not None,
            "model_name": self.model_name,
            "model_display_name": AVAILABLE_MODELS.get(self.model_name, self.model_name),
            "base_url": self.base_url,
            "available_models": list(AVAILABLE_MODELS.keys()),
        }
