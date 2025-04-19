"""
Embedding generation module.

This module handles generating embeddings for text using OpenAI's API.
It includes caching to avoid regenerating embeddings for the same text.
"""

import os
import logging
import time
import json
import hashlib
from typing import List, Dict, Any, Optional

import openai

# Import configurations
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import config
from core.api_manager import rate_limiter

logger = logging.getLogger(__name__)

# Embedding cache to avoid regenerating embeddings for the same text
_embedding_cache = {}
_embedding_cache_file = os.path.join(str(config.MEMORY_DIR), "embedding_cache.json")
_MAX_CACHE_SIZE = 10000  # Maximum number of entries in the cache

def _load_embedding_cache() -> None:
    """Load the embedding cache from disk if it exists."""
    global _embedding_cache
    try:
        if os.path.exists(_embedding_cache_file):
            with open(_embedding_cache_file, 'r') as f:
                _embedding_cache = json.load(f)
                logger.info(f"Loaded {len(_embedding_cache)} embeddings from cache")
    except Exception as e:
        logger.error(f"Error loading embedding cache: {e}")
        _embedding_cache = {}

def _save_embedding_cache() -> None:
    """Save the embedding cache to disk."""
    try:
        # Ensure memory directory exists
        os.makedirs(os.path.dirname(_embedding_cache_file), exist_ok=True)
        
        # If the cache is too large, trim it down
        global _embedding_cache
        if len(_embedding_cache) > _MAX_CACHE_SIZE:
            logger.info(f"Trimming embedding cache from {len(_embedding_cache)} to {_MAX_CACHE_SIZE} entries")
            # Convert to list of tuples, sort by timestamp, and keep the most recent
            cache_items = list(_embedding_cache.items())
            cache_items.sort(key=lambda x: x[1].get('timestamp', 0), reverse=True)
            _embedding_cache = dict(cache_items[:_MAX_CACHE_SIZE])
        
        with open(_embedding_cache_file, 'w') as f:
            json.dump(_embedding_cache, f)
        logger.debug(f"Saved {len(_embedding_cache)} embeddings to cache")
    except Exception as e:
        logger.error(f"Error saving embedding cache: {e}")

# Load the cache when the module is imported
_load_embedding_cache()

def get_embedding(text: str) -> List[float]:
    """
    Get the embedding for a text string.
    Uses caching to avoid regenerating embeddings for the same text.
    
    Args:
        text: The text to embed
        
    Returns:
        List of float values representing the embedding
    """
    if not text or not text.strip():
        logger.warning("Attempted to get embedding for empty text")
        # Return a zero vector of the expected dimension
        return [0.0] * config.EMBEDDING_DIMENSION
    
    # Hash the text to use as a cache key
    text_hash = hashlib.md5(text.encode()).hexdigest()
    
    # Check if we have this embedding in cache
    if text_hash in _embedding_cache:
        cache_entry = _embedding_cache[text_hash]
        # Update the timestamp to mark it as recently used
        cache_entry['timestamp'] = time.time()
        return cache_entry['embedding']
    
    # Generate new embedding
    try:
        # Apply rate limiting
        with rate_limiter('embedding'):
            # Use OpenAI's embedding API
            embedding_model = os.getenv("EMBEDDING_MODEL", config.EMBEDDING_MODEL)
            response = openai.Embedding.create(
                input=text,
                model=embedding_model
            )
            embedding = response['data'][0]['embedding']
            
            # Cache the result
            _embedding_cache[text_hash] = {
                'embedding': embedding,
                'timestamp': time.time()
            }
            
            # Periodically save the cache
            if len(_embedding_cache) % 100 == 0:
                _save_embedding_cache()
                
            return embedding
    except Exception as e:
        logger.error(f"Error generating embedding: {e}")
        # Return a zero vector in case of error
        return [0.0] * config.EMBEDDING_DIMENSION

def batch_get_embeddings(texts: List[str]) -> List[List[float]]:
    """
    Get embeddings for multiple texts in an efficient batch.
    
    Args:
        texts: List of texts to get embeddings for
        
    Returns:
        List of embeddings (each is a list of floats)
    """
    if not texts:
        return []
        
    # Filter out texts that are already in the cache
    texts_to_embed = []
    text_hashes = []
    cached_embeddings = {}
    
    for text in texts:
        if not text or not text.strip():
            # Handle empty text
            cached_embeddings[text] = [0.0] * config.EMBEDDING_DIMENSION
            continue
            
        text_hash = hashlib.md5(text.encode()).hexdigest()
        if text_hash in _embedding_cache:
            # Update timestamp for recently used
            _embedding_cache[text_hash]['timestamp'] = time.time()
            cached_embeddings[text] = _embedding_cache[text_hash]['embedding']
        else:
            texts_to_embed.append(text)
            text_hashes.append(text_hash)
    
    # If all were cached, return the cached values
    if not texts_to_embed:
        return [cached_embeddings[text] for text in texts]
    
    # Generate new embeddings for the remaining texts
    try:
        with rate_limiter('embedding'):
            embedding_model = os.getenv("EMBEDDING_MODEL", config.EMBEDDING_MODEL)
            response = openai.Embedding.create(
                input=texts_to_embed,
                model=embedding_model
            )
            
            # Process and cache the results
            for i, embedding_data in enumerate(response['data']):
                text = texts_to_embed[i]
                text_hash = text_hashes[i]
                embedding = embedding_data['embedding']
                
                # Cache the result
                _embedding_cache[text_hash] = {
                    'embedding': embedding,
                    'timestamp': time.time()
                }
                cached_embeddings[text] = embedding
            
            # Save the cache
            _save_embedding_cache()
    except Exception as e:
        logger.error(f"Error generating batch embeddings: {e}")
        # Fill in zeros for any texts that failed
        for i, text in enumerate(texts_to_embed):
            if text not in cached_embeddings:
                cached_embeddings[text] = [0.0] * config.EMBEDDING_DIMENSION
    
    # Return the embeddings in the same order as the input texts
    return [cached_embeddings.get(text, [0.0] * config.EMBEDDING_DIMENSION) for text in texts] 