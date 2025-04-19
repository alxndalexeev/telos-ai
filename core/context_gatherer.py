import os
import json
import logging
import time
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
from pathlib import Path
import ast
import pinecone
import openai
import config

logger = logging.getLogger(__name__)

# Constants
CACHE_FILE = os.path.join(str(config.MEMORY_DIR), "vector_index_cache.json")
CACHE_TTL_DAYS = 30  # Maximum age of cache entries before forced refresh
MAX_EMBEDDING_BATCH_SIZE = 8  # Maximum number of texts to embed in a single API call
EMBEDDING_RETRY_ATTEMPTS = 3  # Number of retries for embedding API calls
EMBEDDING_RETRY_DELAY = 2  # Seconds to wait between retries

# Use config.py for all Pinecone and embedding model settings
# Example: config.PINECONE_API_KEY, config.PINECONE_INDEX_NAME, etc.

# --- Utility: Get all files in project (with optional max depth) ---
def get_file_tree(root_dir: str, max_depth: int = 3) -> List[str]:
    """Recursively list files in the project up to max_depth."""
    file_list = []
    root_path = Path(root_dir)
    for dirpath, _, filenames in os.walk(root_path):
        try:
            depth = len(Path(dirpath).relative_to(root_path).parts)
            if depth > max_depth:
                continue
            for fname in filenames:
                rel_path = str(Path(dirpath) / fname)
                file_list.append(os.path.relpath(rel_path, root_dir))
        except Exception as e:
            logger.debug(f"Error traversing directory {dirpath}: {e}")
    logger.debug(f"File tree (max_depth={max_depth}): {len(file_list)} files found.")
    return file_list

# --- Utility: Get recently changed files (by mtime) ---
def get_recent_changes(root_dir: str, days: int = 3) -> List[str]:
    """List files changed in the last N days."""
    recent_files = []
    cutoff = datetime.now() - timedelta(days=days)
    for dirpath, _, filenames in os.walk(root_dir):
        for fname in filenames:
            fpath = os.path.join(dirpath, fname)
            try:
                mtime = datetime.fromtimestamp(os.path.getmtime(fpath))
                if mtime > cutoff:
                    recent_files.append(os.path.relpath(fpath, root_dir))
            except Exception as e:
                logger.debug(f"Could not stat file {fpath}: {e}")
    logger.debug(f"Recent changes (last {days} days): {len(recent_files)} files.")
    return recent_files

# --- Utility: Summarize last N lines of a log file ---
def summarize_log(log_file: str, n: int = 10) -> List[str]:
    """Return the last N non-header lines from a log file."""
    if not os.path.exists(log_file):
        logger.info(f"Log file not found: {log_file}")
        return []
    try:
        with open(log_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        content_lines = [l for l in lines if not l.startswith('#') and l.strip()]
        logger.debug(f"Summarized {log_file}: {len(content_lines)} lines, returning last {n}.")
        return content_lines[-n:]
    except Exception as e:
        logger.error(f"Failed to summarize log {log_file}: {e}")
        return []

# --- Utility: Extract open questions/TODOs from memory files ---
def extract_open_questions(memory_dir: str) -> List[str]:
    """Extract lines containing TODO or 'open question' from memory .md files."""
    questions = []
    try:
        for fname in os.listdir(memory_dir):
            if fname.endswith('.md'):
                fpath = os.path.join(memory_dir, fname)
                try:
                    with open(fpath, 'r', encoding='utf-8') as f:
                        for line in f:
                            if 'TODO' in line or 'open question' in line.lower():
                                questions.append(f"{fname}: {line.strip()}")
                except Exception as e:
                    logger.debug(f"Could not read {fpath}: {e}")
        logger.debug(f"Extracted {len(questions)} open questions/TODOs.")
        return questions
    except Exception as e:
        logger.error(f"Failed to extract open questions: {e}")
        return []

# --- Semantic Chunking for Python Code ---
def extract_python_chunks(file_path: str) -> List[Dict[str, Any]]:
    """
    Extract function and class definitions (with docstrings and code) from a Python file.
    
    Returns a list of chunk dictionaries, each containing:
    - text: The extracted code/docstring
    - file: Source file path
    - type: 'code'
    - name: Function/class name
    - start_line: Starting line number
    - end_line: Ending line number
    - timestamp: File modification time
    """
    chunks = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            source = f.read()
        tree = ast.parse(source)
        mtime = os.path.getmtime(file_path)
        
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                start_line = node.lineno
                end_line = getattr(node, 'end_lineno', None) or start_line
                
                # Extract the code lines
                code_lines = source.splitlines()[start_line-1:end_line]
                code = '\n'.join(code_lines)
                
                # Extract docstring if present
                docstring = ast.get_docstring(node) or ""
                
                # Combine name, docstring, and code
                chunk_text = f"{node.name}\n{docstring}\n{code}"
                
                chunks.append({
                    "text": chunk_text,
                    "file": file_path,
                    "type": "code",
                    "name": node.name,
                    "start_line": start_line,
                    "end_line": end_line,
                    "timestamp": mtime
                })
    except Exception as e:
        logger.error(f"Failed to extract code chunks from {file_path}: {e}")
    
    return chunks

# --- Semantic Chunking for Markdown/Docs ---
def extract_markdown_chunks(file_path: str) -> List[Dict[str, Any]]:
    """
    Split Markdown file by headings (##, #, etc.) and add timestamp metadata.
    
    Returns a list of chunk dictionaries, each containing:
    - text: The markdown content
    - file: Source file path
    - type: 'doc'
    - heading: Section heading
    - timestamp: File modification time
    """
    chunks = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        mtime = os.path.getmtime(file_path)
        current_chunk = []
        current_heading = None
        
        for line in lines:
            if line.strip().startswith('#'):
                # Save previous chunk if it exists
                if current_chunk:
                    chunks.append({
                        "text": ''.join(current_chunk),
                        "file": file_path,
                        "type": "doc",
                        "heading": current_heading,
                        "timestamp": mtime
                    })
                    current_chunk = []
                # Set new heading
                current_heading = line.strip()
            
            # Add line to current chunk
            current_chunk.append(line)
        
        # Don't forget the last chunk
        if current_chunk:
            chunks.append({
                "text": ''.join(current_chunk),
                "file": file_path,
                "type": "doc",
                "heading": current_heading,
                "timestamp": mtime
            })
    except Exception as e:
        logger.error(f"Failed to extract markdown chunks from {file_path}: {e}")
    
    return chunks

# --- Pinecone: Initialize and get index ---
def get_pinecone_index() -> Optional[pinecone.Index]:
    """Initialize and return the Pinecone index, creating it if needed."""
    if not config.PINECONE_API_KEY:
        logger.warning("Pinecone API key not set. Semantic search will be disabled.")
        return None
    
    try:
        pc = pinecone.Pinecone(api_key=config.PINECONE_API_KEY)
        
        # Check if index exists, create if not
        if config.PINECONE_INDEX_NAME not in [idx.name for idx in pc.list_indexes()]:
            logger.info(f"Creating new Pinecone index: {config.PINECONE_INDEX_NAME}")
            pc.create_index(
                name=config.PINECONE_INDEX_NAME,
                dimension=config.EMBEDDING_DIM,
                metric="cosine",
                spec=pinecone.ServerlessSpec(cloud="gcp", region="us-central1")
            )
        
        return pc.Index(config.PINECONE_INDEX_NAME)
    except Exception as e:
        logger.error(f"Failed to initialize Pinecone index: {e}")
        return None

# --- OpenAI: Get embedding for text ---
def get_embeddings_batch(texts: List[str]) -> List[List[float]]:
    """
    Get OpenAI embeddings for a batch of texts with retry logic and error handling.
    
    Args:
        texts: List of texts to embed
        
    Returns:
        List of embedding vectors (or empty lists for failed embeddings)
    """
    if not config.OPENAI_API_KEY:
        logger.warning("OpenAI API key not set. Embedding generation will be disabled.")
        return [[] for _ in texts]
    
    # Truncate texts if needed
    truncated_texts = []
    for i, text in enumerate(texts):
        if len(text) > 8000:
            logger.warning(f"Text {i} truncated for embedding (>8000 chars)")
            truncated_texts.append(text[:8000])
        else:
            truncated_texts.append(text)
    
    # Try to get embeddings with retries
    for attempt in range(EMBEDDING_RETRY_ATTEMPTS):
        try:
            openai.api_key = config.OPENAI_API_KEY
            resp = openai.embeddings.create(
                input=truncated_texts, 
                model=config.EMBEDDING_MODEL
            )
            
            # Extract and return embedding vectors in the same order
            embeddings = []
            for i in range(len(texts)):
                embeddings.append(resp.data[i].embedding)
            
            return embeddings
            
        except openai.RateLimitError:
            # Handle rate limit errors with exponential backoff
            wait_time = (2 ** attempt) * EMBEDDING_RETRY_DELAY
            logger.warning(f"Rate limit exceeded for embeddings. Retrying in {wait_time} seconds.")
            time.sleep(wait_time)
        except Exception as e:
            logger.error(f"Failed to get embeddings (attempt {attempt+1}): {e}")
            if attempt < EMBEDDING_RETRY_ATTEMPTS - 1:
                time.sleep(EMBEDDING_RETRY_DELAY)
            else:
                # On final attempt, return empty embeddings
                logger.error("All embedding attempts failed. Returning empty embeddings.")
                return [[] for _ in texts]
    
    # Fallback in case all retries fail
    return [[] for _ in texts]

def get_embedding(text: str) -> List[float]:
    """Get OpenAI embedding for the given text (single text wrapper around batch function)."""
    embeddings = get_embeddings_batch([text])
    return embeddings[0] if embeddings else []

def load_index_cache() -> Dict[str, float]:
    """
    Load the vector index cache mapping chunk IDs to last indexed timestamps.
    
    Returns:
        Dictionary mapping chunk IDs to their last indexed timestamps
    """
    if not os.path.exists(CACHE_FILE):
        logger.info(f"Vector index cache not found at {CACHE_FILE}. Will create new cache.")
        return {}
    
    try:
        with open(CACHE_FILE, 'r', encoding='utf-8') as f:
            cache = json.load(f)
        logger.info(f"Loaded vector index cache: {len(cache)} entries")
        return cache
    except Exception as e:
        logger.warning(f"Could not load vector index cache: {e}. Will do full re-index.")
        return {}

def save_index_cache(cache: Dict[str, float]):
    """Save the vector index cache to disk."""
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(CACHE_FILE), exist_ok=True)
        
        with open(CACHE_FILE, 'w', encoding='utf-8') as f:
            json.dump(cache, f)
        
        logger.info(f"Saved vector index cache: {len(cache)} entries")
    except Exception as e:
        logger.error(f"Failed to save vector index cache: {e}")

def prune_cache(cache: Dict[str, float]) -> Dict[str, float]:
    """
    Remove cache entries that refer to files that no longer exist or are too old.
    
    Args:
        cache: The current cache dictionary
        
    Returns:
        Pruned cache dictionary
    """
    pruned_cache = {}
    now = time.time()
    max_age = CACHE_TTL_DAYS * 24 * 60 * 60  # Convert days to seconds
    
    # Count stats for logging
    expired = 0
    missing = 0
    kept = 0
    
    for chunk_id, timestamp in cache.items():
        # Check if entry is too old
        if now - timestamp > max_age:
            expired += 1
            continue
            
        # Check if file still exists
        try:
            # Extract file path from chunk_id
            parts = chunk_id.split(":")
            if len(parts) >= 1:
                file_part = parts[0]
                # Handle both project files and memory files
                if file_part.endswith('.md'):
                    file_path = os.path.join(str(config.MEMORY_DIR), file_part)
                else:
                    # Try to find in project base dir
                    file_path = None
                    for dirpath, _, filenames in os.walk(str(config.BASE_DIR)):
                        if file_part in filenames:
                            file_path = os.path.join(dirpath, file_part)
                            break
                
                # If file not found, skip this entry
                if not file_path or not os.path.exists(file_path):
                    missing += 1
                    continue
        except Exception:
            # If parsing fails, keep the entry to be safe
            pass
        
        # Keep valid entries
        pruned_cache[chunk_id] = timestamp
        kept += 1
    
    logger.info(f"Pruned cache: kept {kept}, removed {expired} expired and {missing} missing file entries")
    return pruned_cache

def batch_upsert(index: pinecone.Index, vectors: List[Dict], namespace: str, batch_size: int = 100) -> bool:
    """
    Batch upsert vectors to Pinecone with error handling.
    
    Args:
        index: Pinecone index
        vectors: List of vector dictionaries to upsert
        namespace: Pinecone namespace
        batch_size: Maximum vectors per batch
        
    Returns:
        Success status (True if all upserts successful)
    """
    if not vectors:
        return True
    
    success = True
    for i in range(0, len(vectors), batch_size):
        batch = vectors[i:i+batch_size]
        try:
            index.upsert(vectors=batch, namespace=namespace)
        except Exception as e:
            logger.error(f"Failed to upsert batch {i//batch_size}: {e}")
            success = False
    
    return success

# --- Self-updating Vector DB: Incremental update ---
def self_update_vector_db(batch_size: int = 10, force_refresh: bool = False):
    """
    Incrementally update the vector DB: only re-index code, docs, and memory chunks that have changed since last indexed.
    Maintains a local cache of last indexed timestamps for each chunk.
    
    Args:
        batch_size: Number of vectors to upsert in a single batch
        force_refresh: If True, re-index all items regardless of cache
    """
    start_time = time.time()
    index = get_pinecone_index()
    if not index:
        logger.error("Cannot update vector DB: failed to initialize Pinecone index")
        return
    
    # Load and prune cache
    cache = {} if force_refresh else load_index_cache()
    if not force_refresh:
        cache = prune_cache(cache)
    
    updated_cache = cache.copy()
    vectors = []
    pending_texts = []
    pending_chunk_ids = []
    pending_metas = []
    
    updated = 0
    skipped = 0
    
    # Process Python files
    logger.info("Scanning Python files for indexing...")
    for dirpath, _, filenames in os.walk(str(config.BASE_DIR)):
        for fname in filenames:
            if fname.endswith('.py'):
                fpath = os.path.join(dirpath, fname)
                for chunk in extract_python_chunks(fpath):
                    chunk_id = f"{fname}:{chunk.get('name', 'unknown')}:{chunk.get('start_line', 0)}"
                    mtime = chunk['timestamp']
                    
                    # Skip if already indexed and not force refreshing
                    if not force_refresh and cache.get(chunk_id) == mtime:
                        skipped += 1
                        continue
                    
                    # Add to pending batch
                    pending_texts.append(chunk['text'])
                    pending_chunk_ids.append(chunk_id)
                    meta = chunk.copy()
                    meta.pop('text')
                    pending_metas.append(meta)
                    updated_cache[chunk_id] = mtime
                    updated += 1
                    
                    # Process batch when it reaches MAX_EMBEDDING_BATCH_SIZE
                    if len(pending_texts) >= MAX_EMBEDDING_BATCH_SIZE:
                        embeddings = get_embeddings_batch(pending_texts)
                        for i, embedding in enumerate(embeddings):
                            if embedding:  # Skip empty embeddings
                                vectors.append({
                                    "id": pending_chunk_ids[i],
                                    "values": embedding,
                                    "metadata": pending_metas[i]
                                })
                        
                        # Clear pending lists
                        pending_texts = []
                        pending_chunk_ids = []
                        pending_metas = []
                    
                    # Upsert when vectors batch is full
                    if len(vectors) >= batch_size:
                        batch_upsert(index, vectors, config.PINECONE_NAMESPACE, batch_size)
                        vectors = []
    
    # Process Markdown/docs
    logger.info("Scanning documentation files for indexing...")
    for dirpath, _, filenames in os.walk(str(config.BASE_DIR)):
        for fname in filenames:
            if fname.endswith('.md') or fname.endswith('.rst') or fname.endswith('.txt'):
                fpath = os.path.join(dirpath, fname)
                for chunk in extract_markdown_chunks(fpath):
                    chunk_id = f"{fname}:{chunk.get('heading', 'section')}"
                    mtime = chunk['timestamp']
                    
                    # Skip if already indexed and not force refreshing
                    if not force_refresh and cache.get(chunk_id) == mtime:
                        skipped += 1
                        continue
                    
                    # Add to pending batch
                    pending_texts.append(chunk['text'])
                    pending_chunk_ids.append(chunk_id)
                    meta = chunk.copy()
                    meta.pop('text')
                    pending_metas.append(meta)
                    updated_cache[chunk_id] = mtime
                    updated += 1
                    
                    # Process batch when it reaches MAX_EMBEDDING_BATCH_SIZE
                    if len(pending_texts) >= MAX_EMBEDDING_BATCH_SIZE:
                        embeddings = get_embeddings_batch(pending_texts)
                        for i, embedding in enumerate(embeddings):
                            if embedding:  # Skip empty embeddings
                                vectors.append({
                                    "id": pending_chunk_ids[i],
                                    "values": embedding,
                                    "metadata": pending_metas[i]
                                })
                        
                        # Clear pending lists
                        pending_texts = []
                        pending_chunk_ids = []
                        pending_metas = []
                    
                    # Upsert when vectors batch is full
                    if len(vectors) >= batch_size:
                        batch_upsert(index, vectors, config.PINECONE_NAMESPACE, batch_size)
                        vectors = []
    
    # Process memory files
    logger.info("Scanning memory files for indexing...")
    for fname in os.listdir(str(config.MEMORY_DIR)):
        if fname.endswith('.md'):
            fpath = os.path.join(str(config.MEMORY_DIR), fname)
            for chunk in extract_markdown_chunks(fpath):
                chunk_id = f"{fname}:{chunk.get('heading', 'section')}"
                mtime = chunk['timestamp']
                
                # Skip if already indexed and not force refreshing
                if not force_refresh and cache.get(chunk_id) == mtime:
                    skipped += 1
                    continue
                
                # Add to pending batch
                pending_texts.append(chunk['text'])
                pending_chunk_ids.append(chunk_id)
                meta = chunk.copy()
                meta.pop('text')
                pending_metas.append(meta)
                updated_cache[chunk_id] = mtime
                updated += 1
                
                # Process batch when it reaches MAX_EMBEDDING_BATCH_SIZE
                if len(pending_texts) >= MAX_EMBEDDING_BATCH_SIZE:
                    embeddings = get_embeddings_batch(pending_texts)
                    for i, embedding in enumerate(embeddings):
                        if embedding:  # Skip empty embeddings
                            vectors.append({
                                "id": pending_chunk_ids[i],
                                "values": embedding,
                                "metadata": pending_metas[i]
                            })
                    
                    # Clear pending lists
                    pending_texts = []
                    pending_chunk_ids = []
                    pending_metas = []
                
                # Upsert when vectors batch is full
                if len(vectors) >= batch_size:
                    batch_upsert(index, vectors, config.PINECONE_NAMESPACE, batch_size)
                    vectors = []
    
    # Process any remaining pending items
    if pending_texts:
        embeddings = get_embeddings_batch(pending_texts)
        for i, embedding in enumerate(embeddings):
            if embedding:  # Skip empty embeddings
                vectors.append({
                    "id": pending_chunk_ids[i],
                    "values": embedding,
                    "metadata": pending_metas[i]
                })
    
    # Upsert any remaining vectors
    if vectors:
        batch_upsert(index, vectors, config.PINECONE_NAMESPACE, batch_size)
    
    # Save the updated cache
    save_index_cache(updated_cache)
    
    # Log completion info
    elapsed = time.time() - start_time
    mode = "full refresh" if force_refresh else "incremental update"
    logger.info(f"Vector DB {mode} complete in {elapsed:.2f}s. {updated} chunks updated, {skipped} unchanged.")

# --- Pinecone: Semantic search ---
def semantic_search(query: str, top_k: int = 5) -> List[Dict[str, Any]]:
    """
    Semantic search in Pinecone for the most relevant memory items.
    
    Args:
        query: The search query text
        top_k: Number of results to return
        
    Returns:
        List of metadata for matching chunks
    """
    index = get_pinecone_index()
    if not index:
        logger.error("Cannot perform semantic search: failed to initialize Pinecone index")
        return []
    
    embedding = get_embedding(query)
    if not embedding:
        logger.error("Cannot perform semantic search: failed to get embedding for query")
        return []
    
    try:
        results = index.query(
            namespace=config.PINECONE_NAMESPACE,
            vector=embedding,
            top_k=top_k,
            include_values=False,
            include_metadata=True
        )
        logger.debug(f"Semantic search for '{query}': {len(results.matches)} matches.")
        
        # Extract and return metadata
        return [match.metadata for match in results.matches]
    except Exception as e:
        logger.error(f"Semantic search failed: {e}")
        return []

# --- Unified context gathering ---
def gather_context(task: Optional[Dict[str, Any]] = None, top_k: int = 5) -> Dict[str, Any]:
    """
    Gather a rich context for planning and decision making.
    Returns a dict with file tree, recent changes, log/thoughts summaries, open questions, and semantic context.
    TODO: Add codebase semantic indexing for even richer context.
    
    Args:
        task: Current task dictionary
        top_k: Number of semantic results to include
        
    Returns:
        Dictionary with rich context for planning and decision making
    """
    context = {}
    context['file_tree'] = get_file_tree(str(config.BASE_DIR), max_depth=3)
    context['recent_changes'] = get_recent_changes(str(config.BASE_DIR), days=3)
    context['log_summary'] = summarize_log(str(config.ACTION_LOG), n=10)
    context['thoughts_summary'] = summarize_log(str(config.THOUGHTS_LOG), n=10)
    context['open_questions'] = extract_open_questions(str(config.MEMORY_DIR))
    if task:
        query = f"{task.get('task', '')} {task.get('details', '')}"
        context['semantic_context'] = semantic_search(query, top_k=top_k)
    else:
        context['semantic_context'] = []
    logger.info(f"Context gathered for task: {task.get('task') if task else 'N/A'}")
    return context

# --- (Optional) Script to re-index all memory files ---
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Self-update vector DB with code, docs, and memory")
    parser.add_argument('--force', action='store_true', help='Force refresh all items regardless of cache')
    args = parser.parse_args()
    
    if args.force:
        print("Self-updating vector DB with code, docs, and memory (FULL REFRESH)...")
    else:
        print("Self-updating vector DB with code, docs, and memory (incremental)...")
    
    self_update_vector_db(force_refresh=args.force)
    print("Done.") 