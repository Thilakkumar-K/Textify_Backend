"""
Session Manager for Temporary Chat
Handles in-memory storage of temporary documents and their embeddings
"""

import logging
import time
import asyncio
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import uuid

logger = logging.getLogger(__name__)


class TemporarySession:
    """Represents a temporary chat session with in-memory storage"""

    def __init__(self, session_id: str, ttl_minutes: int = 60):
        self.session_id = session_id
        self.created_at = time.time()
        self.expires_at = time.time() + (ttl_minutes * 60)
        self.ttl_minutes = ttl_minutes

        # In-memory storage
        self.documents: Dict[str, Dict[str, Any]] = {}  # doc_id -> document metadata
        self.embeddings: Dict[str, Any] = {}  # doc_id -> embeddings data
        self.chunks: Dict[str, List[Dict[str, Any]]] = {}  # doc_id -> chunks

        self.last_accessed = time.time()

    def is_expired(self) -> bool:
        """Check if session has expired"""
        return time.time() > self.expires_at

    def extend_ttl(self, minutes: int = 30):
        """Extend session TTL"""
        self.expires_at = time.time() + (minutes * 60)
        self.last_accessed = time.time()

    def add_document(self, doc_id: str, filename: str, chunks: List[Dict[str, Any]],
                     embeddings: Any, metadata: Dict[str, Any]):
        """Add document to session"""
        self.documents[doc_id] = {
            "document_id": doc_id,
            "filename": filename,
            "chunks_count": len(chunks),
            "uploaded_at": time.time(),
            "metadata": metadata
        }
        self.chunks[doc_id] = chunks
        self.embeddings[doc_id] = embeddings
        self.last_accessed = time.time()

    def get_document(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """Get document data"""
        self.last_accessed = time.time()
        if doc_id in self.documents:
            return {
                "metadata": self.documents[doc_id],
                "chunks": self.chunks[doc_id],
                "embeddings": self.embeddings[doc_id]
            }
        return None

    def delete_document(self, doc_id: str) -> bool:
        """Delete document from session"""
        self.last_accessed = time.time()
        if doc_id in self.documents:
            del self.documents[doc_id]
            del self.chunks[doc_id]
            del self.embeddings[doc_id]
            return True
        return False

    def list_documents(self) -> List[Dict[str, Any]]:
        """List all documents in session"""
        self.last_accessed = time.time()
        return list(self.documents.values())

    def get_stats(self) -> Dict[str, Any]:
        """Get session statistics"""
        return {
            "session_id": self.session_id,
            "created_at": self.created_at,
            "expires_at": self.expires_at,
            "last_accessed": self.last_accessed,
            "ttl_minutes": self.ttl_minutes,
            "total_documents": len(self.documents),
            "total_chunks": sum(len(chunks) for chunks in self.chunks.values()),
            "time_remaining_seconds": max(0, int(self.expires_at - time.time()))
        }


class SessionManager:
    """Manages temporary chat sessions"""

    def __init__(self, cleanup_interval: int = 300):
        self.sessions: Dict[str, TemporarySession] = {}
        self.cleanup_interval = cleanup_interval  # seconds
        self._cleanup_task = None

    async def start_cleanup_task(self):
        """Start background cleanup task"""
        if self._cleanup_task is None:
            self._cleanup_task = asyncio.create_task(self._periodic_cleanup())
            logger.info("🧹 Session cleanup task started")

    async def _periodic_cleanup(self):
        """Periodically clean up expired sessions"""
        while True:
            try:
                await asyncio.sleep(self.cleanup_interval)
                await self.cleanup_expired_sessions()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in cleanup task: {e}")

    async def cleanup_expired_sessions(self):
        """Remove expired sessions"""
        expired = [sid for sid, session in self.sessions.items() if session.is_expired()]

        for session_id in expired:
            logger.info(f"🗑️ Cleaning up expired session: {session_id}")
            del self.sessions[session_id]

        if expired:
            logger.info(f"✅ Cleaned up {len(expired)} expired sessions")

    def create_session(self, ttl_minutes: int = 60) -> str:
        """Create a new temporary session"""
        session_id = str(uuid.uuid4())
        self.sessions[session_id] = TemporarySession(session_id, ttl_minutes)
        logger.info(f"✅ Created temporary session: {session_id} (TTL: {ttl_minutes}m)")
        return session_id

    def get_session(self, session_id: str) -> Optional[TemporarySession]:
        """Get session by ID"""
        session = self.sessions.get(session_id)

        if session:
            if session.is_expired():
                logger.info(f"⏰ Session expired: {session_id}")
                del self.sessions[session_id]
                return None
            return session

        return None

    def delete_session(self, session_id: str) -> bool:
        """Delete a session"""
        if session_id in self.sessions:
            logger.info(f"🗑️ Deleting session: {session_id}")
            del self.sessions[session_id]
            return True
        return False

    def extend_session(self, session_id: str, minutes: int = 30) -> bool:
        """Extend session TTL"""
        session = self.get_session(session_id)
        if session:
            session.extend_ttl(minutes)
            logger.info(f"⏰ Extended session {session_id} by {minutes} minutes")
            return True
        return False

    def get_all_sessions_stats(self) -> Dict[str, Any]:
        """Get statistics for all sessions"""
        return {
            "total_sessions": len(self.sessions),
            "sessions": [session.get_stats() for session in self.sessions.values()]
        }


# Global instance
_session_manager = None


def get_session_manager() -> SessionManager:
    """Get singleton instance of SessionManager"""
    global _session_manager
    if _session_manager is None:
        _session_manager = SessionManager()
    return _session_manager