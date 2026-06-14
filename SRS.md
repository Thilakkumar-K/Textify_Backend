# Software Requirements Specification (SRS) for RAG-Powered Document QA System

## 1. Introduction

### 1.1 Purpose
This Software Requirements Specification (SRS) documents the functional and non-functional requirements for the RAG-Powered Document QA System. The system provides an intelligent, retrieval-augmented solution for querying documents using Large Language Models (LLMs) via OpenRouter, with Supabase for persistent storage and an in-memory FAISS cache for high-performance retrieval.

### 1.2 Scope
The system encompasses a FastAPI-based backend capable of:
*   Inhaling documents from multiple sources (Uploads, URLs, Google Drive).
*   Extracting and normalizing text from various formats (PDF, DOCX, EML, TXT).
*   Intelligent semantic chunking and vector embedding generation.
*   Hybrid storage: Supabase for persistence and In-Memory FAISS for low-latency search.
*   Question Answering (QA) using RAG across single or multiple documents.
*   Ephemeral, session-based document processing for privacy-sensitive or temporary tasks.

### 1.3 Definitions, Acronyms, and Abbreviations
*   **RAG:** Retrieval-Augmented Generation.
*   **LLM:** Large Language Model (e.g., GPT-4).
*   **FAISS:** Facebook AI Similarity Search (vector indexing library).
*   **Embedding:** A numerical vector representation of text capturing semantic meaning.
*   **Chunking:** The process of splitting large documents into smaller, meaningful segments.
*   **TTL:** Time-To-Live (used for session expiration).
*   **Supabase:** Backend-as-a-Service providing PostgreSQL and Object Storage.
*   **OpenRouter:** Multi-model LLM gateway providing access to various models via an OpenAI-compatible API.

## 2. Overall Description

### 2.1 Product Perspective
The system acts as a high-performance "knowledge engine" API. It is designed to be consumed by frontends (Web/Mobile) or integrated into larger enterprise workflows. It abstracts the complexity of vector databases and LLM prompt engineering into simple RESTful endpoints.

### 2.2 Product Functions
1.  **Document Ingestion:** Multi-source ingestion with automatic format detection.
2.  **Vector Management:** Automated embedding generation and synchronization between Supabase and In-Memory FAISS.
3.  **Intelligent RAG:** Semantic retrieval followed by LLM-based answer synthesis with source attribution.
4.  **Session Management:** Creation of temporary, in-memory "sandboxes" for document interaction.
5.  **Administrative Tools:** Health monitoring, cache management, and connectivity testing.

### 2.3 User Classes and Characteristics
*   **End Users:** Non-technical users querying documents.
*   **Developers:** Users integrating the API into other platforms.
*   **Administrators:** Users managing system health and costs (API usage).

### 2.4 Operating Environment
*   **Backend:** Python 3.9+ with FastAPI.
*   **Deployment:** Containerized (Docker), optimized for Google Cloud Run.
*   **Storage:** Supabase Storage (Object Storage).
*   **AI Services:** OpenRouter (multi-model access including free and premium models) and Sentence-Transformers (`all-MiniLM-L6-v2`).

### 2.5 General Constraints
*   **Memory:** In-memory caching requires sufficient RAM relative to the number of document chunks.
*   **Security:** Bearer Token authentication required for all sensitive endpoints.
*   **Statelessness:** The backend is stateless; all state is in Supabase or reconstructed in memory on startup.

## 3. System Architecture

### 3.1 Data Flow Diagram
```
[User/Frontend] -> [FastAPI API]
                        |
        +---------------+---------------+
        |               |               |
[Vector Cache]   [Document Processor] [OpenRouter LLM]
(FAISS/RAM)      (NLTK/PyPDF2)        (Multi-model RAG)
        ^               |
        |               v
        +------- [Supabase Storage]
```

### 3.2 Component Details
*   **DocumentProcessor:** Handles `PyPDF2`, `python-docx`, and `email` modules for text extraction. Implements NLTK-based intelligent chunking.
*   **VectorStore/VectorCache:** Uses `SentenceTransformer` for local embedding and `FAISS` (IndexFlatIP) for similarity search.
*   **OpenRouterService:** Manages prompt construction (System/User) and handles the chat completion lifecycle via OpenRouter's OpenAI-compatible API.
*   **SupabaseUtils:** Wraps `supabase-py` for file uploads, downloads, and listing.

## 4. Functional Requirements (Backend)

### 4.1 Document Processing
*   **FR-BP-1:** The system shall support PDF, DOCX, EML, and TXT files.
*   **FR-BP-2:** The system shall implement `Intelligent Chunking` (default: 1500 chars, 200 overlap) using sentence tokenization to avoid splitting sentences.
*   **FR-BP-3:** The system shall generate 384-dimension embeddings using the `all-MiniLM-L6-v2` model.

### 4.2 Storage & Caching
*   **FR-SC-1:** All vectors (`.npy`), chunks (`.json`), and metadata (`.json`) shall be stored in Supabase under `vectors/{doc_id}/`.
*   **FR-SC-2:** On startup or manual refresh, the system shall load all metadata from Supabase into the `VectorCache`.
*   **FR-SC-3:** Similarity search shall be performed using FAISS `IndexFlatIP` (Inner Product) on normalized vectors (effectively Cosine Similarity).

### 4.3 RAG Logic
*   **FR-RL-1:** The system shall retrieve `TOP_K` (default 8) chunks for every question.
*   **FR-RL-2:** The system shall use a conversational persona for the LLM, focusing on Markdown-formatted, friendly responses.
*   **FR-RL-3:** If no relevant context is found (similarity score < threshold), the system shall inform the user rather than hallucinating.

### 4.4 API Endpoints (v1)

#### 4.4.1 Document Management
*   `GET /`: List all available documents (brief metadata).
*   `GET /api/v1/documents`: Detailed list of stored documents.
*   `POST /api/v1/documents/upload`: Upload a file (Multipart).
*   `DELETE /api/v1/documents/{document_id}`: Delete from Supabase and Cache.

#### 4.4.2 Question Answering
*   `POST /api/v1/hackrx/run`: QA for a specific document (URL or ID).
    *   **Request:** `{ "documents": "URL", "questions": ["..."], "document_id": "..." }`
*   `POST /api/v1/query/global`: Global search across multiple documents.
    *   **Request:** `{ "query": "...", "document_ids": ["..."], "top_k": 10 }`

#### 4.4.3 Temporary Sessions
*   `POST /api/v1/temporary/session/create`: Create an ephemeral session.
*   `POST /api/v1/temporary/upload`: Upload to session (Memory Only).
*   `POST /api/v1/temporary/query`: Query session-specific documents.

#### 4.4.4 System Admin
*   `GET /api/v1/health`: Detailed health and cache stats.
*   `POST /api/v1/cache/refresh`: Clear and reload from Supabase.
*   `GET /api/v1/test-llm`: LLM connectivity check.
*   `GET /api/v1/models`: List available OpenRouter models.
*   `POST /api/v1/models/switch`: Hot-swap active model.

## 5. Functional Requirements (Frontend - Placeholders)

### 5.1 User Interface (UI)
*   **PH-FE-UI-1 (Dashboard):** A clean dashboard showing a list of uploaded documents with status indicators.
*   **PH-FE-UI-2 (Chat Interface):** A modern chat window supporting Markdown rendering for AI responses.
*   **PH-FE-UI-3 (Upload Modal):** Support for drag-and-drop and URL input.
*   **PH-FE-UI-4 (Global Search):** A unified search bar to query all documents simultaneously.

### 5.2 Interactions (UX)
*   **PH-FE-UX-1:** Real-time loading states for document ingestion and QA.
*   **PH-FE-UX-2:** Side-by-side view (optional) to see source chunks alongside the answer.
*   **PH-FE-UX-3:** Error notifications for failed uploads or invalid queries.

### 5.3 Technical Implementation
*   **PH-FE-TECH-1:** Implementation using React, Angular, or Vue.
*   **PH-FE-TECH-2:** State management (Redux/Zustand) for handling active sessions.
*   **PH-FE-TECH-3:** Secure storage of the API Bearer Token in browser session/local storage.

## 6. Non-Functional Requirements

### 6.1 Performance
*   **NFR-1:** Retrieval latency (FAISS) shall be < 100ms for 10,000 chunks.
*   **NFR-2:** End-to-end QA latency (including LLM) should be < 5s for most queries.

### 6.2 Security
*   **NFR-3:** All communication must be over HTTPS.
*   **NFR-4:** Environment variables must be used for all secrets (OpenRouter/Supabase keys).

### 6.3 Scalability
*   **NFR-5:** The system should support horizontal scaling of the FastAPI layer (keeping in mind that each instance maintains its own in-memory cache).

## 7. Data Model

### 7.1 Supabase Schema (Object Storage)
*   `documents/`: Raw files.
*   `vectors/{doc_id}/metadata.json`: Contains `file_name`, `chunks_count`, `created_at`, `source_info`.
*   `vectors/{doc_id}/chunks.json`: Array of `{ "text": "...", "chunk_id": 0, "char_count": 0 }`.
*   `vectors/{doc_id}/embeddings.npy`: Binary numpy array of shape `(N, 384)`.

## 8. Appendices

### 8.1 Environment Variables
*   `SUPABASE_URL`, `SUPABASE_KEY`: Persistence.
*   `OPENROUTER_API_KEY`, `OPENROUTER_MODEL`: Intelligence.
*   `VALID_TOKEN`: Security.

### 8.2 Future Enhancements
*   Integration with Microsoft Teams/Slack.
*   User-specific document namespaces.
*   Streaming responses from OpenRouter.
