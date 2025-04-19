"""
Vector database management module for Telos AI.

This module handles all vector database operations including:
- Embedding generation
- Index management
- Knowledge storage
- Refresh scheduling
"""

from core.vector_db.index_manager import get_pinecone_index
from core.vector_db.embedding import get_embedding
from core.vector_db.knowledge_store import (
    create_knowledge_chunks,
    store_knowledge_in_vector_db
)
from core.vector_db.refresh import (
    check_refresh_needed,
    record_refresh_timestamp,
    update_vector_db
)

__all__ = [
    'get_pinecone_index',
    'get_embedding',
    'create_knowledge_chunks',
    'store_knowledge_in_vector_db',
    'check_refresh_needed',
    'record_refresh_timestamp',
    'update_vector_db'
]
