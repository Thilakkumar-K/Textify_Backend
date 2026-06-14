# RAG-Powered Document QA API with OpenRouter

A production-ready FastAPI backend for intelligent document question answering using Retrieval-Augmented Generation (RAG), FAISS vector search, OpenRouter LLM, and Supabase storage.

## 🚀 Features

- **Intelligent Document Processing**: Support for PDF, DOCX, and email formats with smart chunking
- **RAG Architecture**: Semantic search with FAISS + OpenRouter LLM generation
- **High-Performance Caching**: In-memory vector cache for instant retrieval
- **Cloud Storage**: Supabase integration for persistent storage (no local storage)
- **Global Search**: Cross-document search capabilities
- **Hot Model Switching**: Switch between OpenRouter models without restarting
- **Production Ready**: Built for Google Cloud Run with graceful shutdown handling
- **Concurrent Processing**: Parallel question processing for maximum throughput

## 📋 Requirements

- Python 3.9+
- OpenRouter API key (get one at https://openrouter.ai/keys)
- Supabase account with storage bucket
- Google Cloud Run (for deployment)

## 🔧 Installation

### 1. Clone the Repository

```bash
git clone <repository-url>
cd rag-document-qa-api
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Environment Variables

Create a `.env` file with the following variables:

```bash
# Authentication
VALID_TOKEN=your_secure_token_here

# OpenRouter Configuration
OPENROUTER_API_KEY=sk-or-your-key-here
OPENROUTER_MODEL=openrouter/free
OPENROUTER_APP_NAME=Microsoft RAG QA

# Supabase Configuration
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_KEY=your_supabase_anon_key
SUPABASE_BUCKET=documents

# Optional
ENVIRONMENT=production
DEBUG=false
```

## 🏃 Running the Application

### Local Development

```bash
python main.py
```

Or with uvicorn:

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

### Docker Deployment

```bash
docker build -t rag-qa-api .
docker run -p 8000:8000 --env-file .env rag-qa-api
```

### Google Cloud Run

```bash
gcloud run deploy rag-qa-api \
  --source . \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated
```

## 📚 API Endpoints

### Health Check

```bash
GET /api/v1/health
```

Returns service status, configuration, and cache statistics.

### Upload Document

```bash
POST /api/v1/documents/upload
Authorization: Bearer YOUR_VALID_TOKEN
Content-Type: multipart/form-data

file: <document.pdf>
```

**Response:**
```json
{
  "document_id": "uuid-here",
  "filename": "document.pdf",
  "status": "processed_and_cached",
  "chunks_created": 42,
  "message": "Document processed successfully",
  "supabase_path": "documents/...",
  "public_url": "https://..."
}
```

### Document Q&A (RAG Endpoint)

```bash
POST /api/v1/hackrx/run
Authorization: Bearer YOUR_VALID_TOKEN
Content-Type: application/json

{
  "documents": "https://example.com/document.pdf",
  "questions": [
    "What is the main topic?",
    "Who are the key stakeholders?"
  ]
}
```

Or using document_id:

```json
{
  "document_id": "existing-doc-uuid",
  "questions": ["Your question here"]
}
```

**Response:**
```json
{
  "answers": [
    {
      "question": "What is the main topic?",
      "answer": "The document discusses...",
      "confidence": 0.95,
      "sources": [...],
      "chunks_retrieved": 8
    }
  ],
  "document_id": "uuid",
  "cache_hit": true,
  "response_format": "markdown"
}
```

### Global Search

```bash
POST /api/v1/query/global
Authorization: Bearer YOUR_VALID_TOKEN
Content-Type: application/json

{
  "query": "What are the key findings?",
  "top_k": 10,
  "max_docs": 5
}
```

**Filtered Search:**
```json
{
  "query": "Your search query",
  "document_ids": ["doc-uuid-1", "doc-uuid-2"],
  "top_k": 10
}
```

### List All Documents

```bash
GET /
# or
GET /api/v1/documents
Authorization: Bearer YOUR_VALID_TOKEN
```

### Delete Document

```bash
DELETE /api/v1/documents/{document_id}
Authorization: Bearer YOUR_VALID_TOKEN
```

### Model Management

```bash
# List available models
GET /api/v1/models

# Switch active model (hot-swap)
POST /api/v1/models/switch?model_id=openai/gpt-4o
Authorization: Bearer YOUR_VALID_TOKEN

# Test LLM connectivity
GET /api/v1/test-llm
```

### Cache Management

```bash
# Get cache statistics
GET /api/v1/cache/stats
Authorization: Bearer YOUR_VALID_TOKEN

# Refresh cache (reload from Supabase)
POST /api/v1/cache/refresh
Authorization: Bearer YOUR_VALID_TOKEN

# Clear cache (memory only)
POST /api/v1/cache/clear
Authorization: Bearer YOUR_VALID_TOKEN
```

## 🏗️ Architecture

### Components

1. **Document Processor**: Extracts text from PDF, DOCX, and email formats
2. **Intelligent Chunking**: Semantic-aware text splitting with overlap
3. **Vector Store**: FAISS-based similarity search with embeddings
4. **In-Memory Cache**: High-performance caching layer for instant retrieval
5. **OpenRouter Service**: LLM-powered answer generation via OpenRouter API
6. **Supabase Storage**: Persistent cloud storage for documents and vectors

### Data Flow

```
Upload → Extract Text → Chunk → Generate Embeddings → Store (Supabase + Cache)
                                                              ↓
Query → Retrieve Vectors (Cache/Supabase) → Search → Generate Answer (OpenRouter)
```

## ⚙️ Configuration

### Chunking Parameters

```python
CHUNK_SIZE = 1500          # Characters per chunk
CHUNK_OVERLAP = 200        # Overlap between chunks
```

### Retrieval Parameters

```python
TOP_K_RETRIEVAL = 8        # Number of chunks to retrieve
MAX_CONTEXT_LENGTH = 10000 # Maximum context for LLM
```

### Embedding Model

```python
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"  # Sentence transformer model
```

## 🔒 Security

- **Token-based Authentication**: All endpoints require Bearer token
- **CORS Enabled**: Configurable cross-origin resource sharing
- **Input Validation**: Pydantic models for request validation
- **Secure Storage**: Supabase with row-level security

## 📊 Performance

- **Caching Strategy**: In-memory cache for instant repeated queries
- **Parallel Processing**: Concurrent question processing
- **Preloading**: Vectors loaded into memory at startup
- **Batch Embeddings**: Efficient batch processing for embeddings

### Typical Response Times

- Cached document query: **< 1 second**
- New document query: **2-5 seconds**
- Document upload + processing: **5-15 seconds** (depends on size)

## 🐛 Debugging

### Check Service Health

```bash
curl http://localhost:8000/api/v1/health
```

### View Logs

```bash
# Local
python main.py

# Cloud Run
gcloud run logs read rag-qa-api --limit 50
```

### Test OpenRouter LLM

```bash
curl http://localhost:8000/api/v1/test-llm
```

## 📝 Example Usage

### Python Client

```python
import requests

API_URL = "http://localhost:8000"
TOKEN = "your_valid_token"

headers = {"Authorization": f"Bearer {TOKEN}"}

# Upload document
with open("document.pdf", "rb") as f:
    response = requests.post(
        f"{API_URL}/api/v1/documents/upload",
        headers=headers,
        files={"file": f}
    )
    doc_id = response.json()["document_id"]

# Ask questions
response = requests.post(
    f"{API_URL}/api/v1/hackrx/run",
    headers=headers,
    json={
        "document_id": doc_id,
        "questions": ["What is this about?"]
    }
)
print(response.json()["answers"][0]["answer"])
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## 📄 License

MIT License - see LICENSE file for details

## 🆘 Support

For issues and questions:
- Create an issue in the repository
- Check existing documentation
- Review API endpoint examples

## 🔄 Version History

- **v3.1.0**: Current version — migrated to OpenRouter with hot model switching
- **v3.0.0**: In-memory caching layer
- **v2.x**: Supabase storage integration
- **v1.x**: Initial RAG implementation

---

**Built with ❤️ using FastAPI, FAISS, OpenRouter, and Supabase**