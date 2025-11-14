"""
Azure OpenAI Service Handler
Handles all interactions with Azure OpenAI API for RAG-based document QA
"""

import logging
import os
from typing import List, Dict, Any, Optional
from openai import AzureOpenAI
import asyncio
from functools import wraps

logger = logging.getLogger(__name__)


def async_wrap(func):
    """Decorator to make sync functions async"""

    @wraps(func)
    async def run(*args, **kwargs):
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, lambda: func(*args, **kwargs))

    return run


class AzureOpenAIService:
    """Service class for Azure OpenAI API interactions"""

    def __init__(
            self,
            api_key: Optional[str] = None,
            endpoint: Optional[str] = None,
            api_version: Optional[str] = None,
            deployment_name: Optional[str] = None
    ):
        """
        Initialize Azure OpenAI Service

        Args:
            api_key: Azure OpenAI API key
            endpoint: Azure OpenAI endpoint URL
            api_version: API version (default: 2024-02-15-preview)
            deployment_name: Deployment/model name
        """
        self.api_key = api_key or os.getenv("AZURE_OPENAI_API_KEY")
        self.endpoint = endpoint or os.getenv("AZURE_OPENAI_ENDPOINT")
        self.api_version = api_version or os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview")
        self.deployment_name = deployment_name or os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4")

        # Validate required credentials
        if not self.api_key:
            raise ValueError("AZURE_OPENAI_API_KEY environment variable is required")
        if not self.endpoint:
            raise ValueError("AZURE_OPENAI_ENDPOINT environment variable is required")

        # Initialize client
        self.client = None
        self.model_name = None
        self._initialize_client()

    def _initialize_client(self):
        """Initialize Azure OpenAI client"""
        try:
            self.client = AzureOpenAI(
                api_key=self.api_key,
                api_version=self.api_version,
                azure_endpoint=self.endpoint
            )

            # Test connection with a simple request
            test_response = self.client.chat.completions.create(
                model=self.deployment_name,
                messages=[{"role": "user", "content": "Test"}],
                max_tokens=10,
                temperature=0.1
            )

            if test_response:
                self.model_name = self.deployment_name
                logger.info(f"✅ Successfully initialized Azure OpenAI with deployment: {self.deployment_name}")

        except Exception as e:
            logger.error(f"❌ Failed to initialize Azure OpenAI client: {e}")
            self.client = None
            self.model_name = None
            raise

    async def generate_rag_answer(
            self,
            question: str,
            relevant_chunks: List[Dict[str, Any]],
            document_id: str,
            temperature: float = 0.3,
            max_tokens: int = 500
    ) -> Dict[str, Any]:
        """
        Generate answer using RAG approach with Azure OpenAI

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
                "answer": "Azure OpenAI service not available",
                "confidence": 0.0,
                "sources": [],
                "chunks_retrieved": 0
            }

        try:
            logger.info(f"🔍 Retrieved {len(relevant_chunks)} chunks for question: {question[:50]}...")

            if not relevant_chunks:
                logger.warning(f"⚠️ No context retrieved for document: {document_id}")
                return {
                    "answer": "No relevant context found in the document to answer this question.",
                    "confidence": 0.0,
                    "sources": [],
                    "chunks_retrieved": 0
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

            # Generate response using Azure OpenAI
            response = await self._generate_completion(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=temperature,
                max_tokens=max_tokens
            )

            if response:
                answer_text = response.strip()
                confidence = self._estimate_confidence(relevant_chunks)

                # Extract sources
                sources = [
                    {
                        "chunk_id": chunk["chunk_id"],
                        "similarity_score": chunk["similarity_score"],
                        "preview": chunk["text"][:200] + "..." if len(chunk["text"]) > 200 else chunk["text"]
                    }
                    for chunk in relevant_chunks[:3]
                ]

                logger.info(f"✅ Generated answer with confidence {confidence:.2f}")

                return {
                    "answer": answer_text,
                    "confidence": confidence,
                    "sources": sources,
                    "chunks_retrieved": len(relevant_chunks),
                    "model_used": self.model_name
                }
            else:
                logger.error("❌ Azure OpenAI returned empty response")
                return {
                    "answer": "Unable to generate response - empty response from Azure OpenAI",
                    "confidence": 0.0,
                    "sources": [],
                    "chunks_retrieved": len(relevant_chunks)
                }

        except Exception as e:
            logger.error(f"❌ Error generating RAG answer: {e}")
            return {
                "answer": f"Error generating answer: {str(e)}",
                "confidence": 0.0,
                "sources": [],
                "chunks_retrieved": len(relevant_chunks)
            }

    async def _generate_completion(
            self,
            system_prompt: str,
            user_prompt: str,
            temperature: float = 0.2,
            max_tokens: int = 500
    ) -> Optional[str]:
        """
        Generate completion using Azure OpenAI Chat API

        Args:
            system_prompt: System message for model behavior
            user_prompt: User message with question and context
            temperature: Sampling temperature
            max_tokens: Maximum tokens in response

        Returns:
            Generated text response or None
        """
        try:
            # Run the synchronous API call in a thread pool
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: self.client.chat.completions.create(
                    model=self.deployment_name,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=temperature,
                    max_tokens=max_tokens,
                    top_p=0.8,
                    frequency_penalty=0.0,
                    presence_penalty=0.0
                )
            )

            if response and response.choices:
                return response.choices[0].message.content
            return None

        except Exception as e:
            logger.error(f"Error calling Azure OpenAI API: {e}")
            raise

    def _build_context(self, chunks: List[Dict[str, Any]], max_length: int = 10000) -> str:
        """
        Build context string from retrieved chunks

        Args:
            chunks: List of document chunks
            max_length: Maximum context length in characters

        Returns:
            Formatted context string
        """
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
        """
        Create system and user prompts for RAG

        Args:
            question: User's question
            context: Retrieved context

        Returns:
            Tuple of (system_prompt, user_prompt)
        """
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
        """
        Estimate confidence based on similarity scores

        Args:
            chunks: Retrieved chunks with similarity scores

        Returns:
            Confidence score between 0 and 1
        """
        if not chunks:
            return 0.0

        # Weighted average of top chunks
        weights = [1.0, 0.8, 0.6, 0.4, 0.2]
        total_score = 0.0
        total_weight = 0.0

        for i, chunk in enumerate(chunks[:5]):
            weight = weights[i] if i < len(weights) else 0.1
            total_score += chunk["similarity_score"] * weight
            total_weight += weight

        confidence = total_score / total_weight if total_weight > 0 else 0.0
        return min(confidence, 1.0)

    async def test_connection(self) -> Dict[str, Any]:
        """
        Test Azure OpenAI connection and functionality

        Returns:
            Dict with test results
        """
        try:
            if not self.client:
                return {
                    "status": "error",
                    "message": "Azure OpenAI client not initialized"
                }

            # Test with a simple RAG-style prompt
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
                max_tokens=100
            )

            return {
                "status": "success",
                "message": "Azure OpenAI API is working with RAG",
                "model_name": self.model_name,
                "deployment_name": self.deployment_name,
                "test_response": response if response else "No response text",
                "endpoint": self.endpoint,
                "api_version": self.api_version
            }

        except Exception as e:
            return {
                "status": "error",
                "message": f"Azure OpenAI test failed: {str(e)}"
            }

    def get_service_info(self) -> Dict[str, Any]:
        """
        Get service configuration information

        Returns:
            Dict with service details
        """
        return {
            "service": "Azure OpenAI",
            "model_available": self.client is not None,
            "model_name": self.model_name,
            "deployment_name": self.deployment_name,
            "endpoint": self.endpoint,
            "api_version": self.api_version
        }