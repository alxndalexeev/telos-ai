"""
Vector database configuration for Telos AI.

This module contains settings related to vector database operations.
"""

import os

# Pinecone index configuration
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "telos-memory")
PINECONE_NAMESPACE = os.getenv("PINECONE_NAMESPACE", "main")

# Embedding configuration
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-ada-002") 
EMBEDDING_DIMENSION = 1536  # Dimension for OpenAI ada-002 embeddings

# Vector DB general settings
USE_VECTOR_DB = True  # Set to False to disable vector database functionality
VECTOR_DB_FULL_REFRESH_DAYS = 1  # How often to do a full refresh (days)
VECTOR_DB_CACHE_SIZE = 10000  # Maximum number of embeddings to cache 