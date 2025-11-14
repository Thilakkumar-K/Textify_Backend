"""
In-Memory Vector Cache Manager
Caches embeddings, chunks, and metadata to avoid repeated Supabase downloads
Thread-safe implementation for production use
"""

import logging
import asyncio
import numpy as np
import faiss
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import threading
import json

logger = logging.getLogger(__name__)


class VectorCache:
    """
    Thread-safe in-memory cache for document vectors and metadata
    Eliminates redundant Supabase downloads
    """

    def __init__(self):
        self._lock = threading.RLock()  # Reentrant lock for thread safety
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._initialized = False
        self._last_refresh = None
        logger.info("🧠 VectorCache initialized")

    def is_cached(self, document_id: str) -> bool:
        """Check if document is already cached"""
        with self._lock:
            return document_id in self._cache

    def get(self, document_id: str) -> Optional[Dict[str, Any]]:
        """
        Get cached document data

        Returns:
            Dict containing:
            - embeddings: np.ndarray
            - chunks: List[Dict]
            - metadata: Dict
            - faiss_index: faiss.Index
            - cached_at: datetime
        """
        with self._lock:
            return self._cache.get(document_id)

    def set(self, document_id: str, embeddings: np.ndarray, chunks: List[Dict[str, Any]],
            metadata: Dict[str, Any]):
        """
        Cache document vectors and metadata

        Args:
            document_id: Unique document identifier
            embeddings: Numpy array of embeddings
            chunks: List of document chunks
            metadata: Document metadata
        """
        with self._lock:
            # Create FAISS index for fast similarity search
            dimension = embeddings.shape[1]
            index = faiss.IndexFlatIP(dimension)

            # Normalize embeddings for cosine similarity
            embeddings_normalized = embeddings.copy().astype('float32')
            faiss.normalize_L2(embeddings_normalized)
            index.add(embeddings_normalized)

            self._cache[document_id] = {
                "embeddings": embeddings,
                "chunks": chunks,
                "metadata": metadata,
                "faiss_index": index,
                "cached_at": datetime.now()
            }

            logger.info(f"✅ Cached document {document_id} ({len(chunks)} chunks, {embeddings.shape[0]} vectors)")

    def get_all_document_ids(self) -> List[str]:
        """Get list of all cached document IDs"""
        with self._lock:
            return list(self._cache.keys())

    def get_all_metadata(self) -> List[Dict[str, Any]]:
        """Get metadata for all cached documents"""
        with self._lock:
            result = []
            for doc_id, data in self._cache.items():
                meta = data["metadata"].copy()
                meta["document_id"] = doc_id
                meta["cached_at"] = data["cached_at"].isoformat()
                result.append(meta)
            return result

    def remove(self, document_id: str) -> bool:
        """Remove document from cache"""
        with self._lock:
            if document_id in self._cache:
                del self._cache[document_id]
                logger.info(f"🗑️ Removed document {document_id} from cache")
                return True
            return False

    def clear(self):
        """Clear entire cache"""
        with self._lock:
            count = len(self._cache)
            self._cache.clear()
            self._initialized = False
            self._last_refresh = None
            logger.info(f"🧹 Cleared cache ({count} documents removed)")

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        with self._lock:
            total_chunks = sum(len(data["chunks"]) for data in self._cache.values())
            total_vectors = sum(data["embeddings"].shape[0] for data in self._cache.values())

            return {
                "total_documents": len(self._cache),
                "total_chunks": total_chunks,
                "total_vectors": total_vectors,
                "initialized": self._initialized,
                "last_refresh": self._last_refresh.isoformat() if self._last_refresh else None,
                "document_ids": list(self._cache.keys())
            }

    def mark_initialized(self):
        """Mark cache as fully initialized"""
        with self._lock:
            self._initialized = True
            self._last_refresh = datetime.now()
            logger.info(f"✅ Cache initialized with {len(self._cache)} documents")

    def is_initialized(self) -> bool:
        """Check if cache has been initialized"""
        with self._lock:
            return self._initialized


# Global singleton instance
_vector_cache_instance: Optional[VectorCache] = None
_cache_lock = threading.Lock()


def get_vector_cache() -> VectorCache:
    """Get singleton VectorCache instance (thread-safe)"""
    global _vector_cache_instance

    if _vector_cache_instance is None:
        with _cache_lock:
            if _vector_cache_instance is None:  # Double-check locking
                _vector_cache_instance = VectorCache()

    return _vector_cache_instance