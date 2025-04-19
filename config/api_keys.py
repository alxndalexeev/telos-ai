"""
API key configuration for Telos AI.

This module contains settings related to API keys and external services.
"""

import os
from dotenv import load_dotenv, find_dotenv

# Load environment variables if not already loaded
env_path = find_dotenv()
if env_path:
    load_dotenv(env_path)

# OpenAI API configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# Tavily API configuration for search
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")

# Pinecone API configuration for vector database
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "")
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT", "gcp-starter") 