"""
Knowledge storage module for vector database.

This module handles creating and storing knowledge chunks in the vector database.
"""

import os
import time
import logging
import json
from typing import List, Dict, Any, Optional

# Import configurations
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import config

# Import from the same package
from core.vector_db.index_manager import get_pinecone_index
from core.vector_db.embedding import get_embedding, batch_get_embeddings

logger = logging.getLogger(__name__)

def create_knowledge_chunks(
    task_name: str, 
    search_results: Optional[str], 
    reviewer_hint: Optional[str], 
    plan: List[Dict[str, Any]], 
    results: Any
) -> List[Dict]:
    """
    Create knowledge chunks for vector DB from task execution data.
    
    Args:
        task_name: Name of the task
        search_results: Results from online search
        reviewer_hint: Reviewer's comments on the plan
        plan: Execution plan
        results: Results from plan execution
        
    Returns:
        List of knowledge chunk dictionaries
    """
    knowledge_chunks = []
    now_ts = time.time()
    
    if search_results:
        knowledge_chunks.append({
            "text": str(search_results),
            "type": "online_search",
            "source": "tavily_search",
            "task": task_name,
            "timestamp": now_ts
        })
    
    if reviewer_hint:
        knowledge_chunks.append({
            "text": reviewer_hint,
            "type": "reviewer_hint",
            "task": task_name,
            "timestamp": now_ts
        })
    
    if plan:
        knowledge_chunks.append({
            "text": json.dumps(plan),
            "type": "plan",
            "task": task_name,
            "timestamp": now_ts
        })
    
    if results:
        knowledge_chunks.append({
            "text": str(results),
            "type": "execution_result",
            "task": task_name,
            "timestamp": now_ts
        })
    
    return knowledge_chunks

def store_knowledge_in_vector_db(
    knowledge_chunks: List[Dict], 
    task_name: str
) -> bool:
    """
    Store knowledge chunks in the vector database.
    
    Args:
        knowledge_chunks: List of knowledge chunks to store
        task_name: Name of the task
        
    Returns:
        True if successful, False otherwise
    """
    if not knowledge_chunks:
        logger.info("No knowledge chunks to store")
        return False
        
    try:
        index = get_pinecone_index()
        if not index:
            logger.warning("Could not get Pinecone index, skipping vector storage")
            return False
            
        vectors = []
        now_ts = time.time()
        
        # Get all embeddings in a single batch for efficiency
        texts = [chunk["text"] for chunk in knowledge_chunks]
        embeddings = batch_get_embeddings(texts)
        
        for i, (chunk, embedding) in enumerate(zip(knowledge_chunks, embeddings)):
            try:
                meta = chunk.copy()
                vectors.append({
                    "id": f"{task_name}:{chunk['type']}:{int(now_ts)}:{i}",
                    "values": embedding,
                    "metadata": meta
                })
            except Exception as e:
                logger.warning(f"Failed to process chunk {i}: {e}")
                continue
                
        if vectors:
            namespace = os.getenv("PINECONE_NAMESPACE", config.PINECONE_NAMESPACE)
            index.upsert(vectors=vectors, namespace=namespace)
            logger.info(f"Upserted {len(vectors)} new knowledge chunks to vector DB")
            return True
        
        logger.warning("No vectors to upsert after processing")
        return False
    except Exception as e:
        logger.error(f"Error storing knowledge in vector DB: {e}")
        return False
    
def delete_knowledge_by_task(task_name: str) -> bool:
    """
    Delete all knowledge chunks related to a specific task.
    
    Args:
        task_name: Name of the task
        
    Returns:
        True if successful, False otherwise
    """
    try:
        index = get_pinecone_index()
        if not index:
            logger.warning("Could not get Pinecone index, skipping deletion")
            return False
        
        namespace = os.getenv("PINECONE_NAMESPACE", config.PINECONE_NAMESPACE)
        
        # Use the metadata filter to find all vectors related to this task
        response = index.delete(
            filter={"task": {"$eq": task_name}},
            namespace=namespace
        )
        
        logger.info(f"Deleted knowledge chunks for task '{task_name}'")
        return True
    except Exception as e:
        logger.error(f"Error deleting knowledge for task '{task_name}': {e}")
        return False 