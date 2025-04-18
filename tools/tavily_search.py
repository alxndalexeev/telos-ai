import os
import requests
from dotenv import load_dotenv
import logging

# Configure logging
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

TAVILY_API_KEY = os.getenv('TAVILY_API_KEY')
if not TAVILY_API_KEY:
    logger.warning("TAVILY_API_KEY not found in environment variables.")

def run_search(query, max_results=5):
    """Perform a web search using the Tavily API and return results."""
    if not TAVILY_API_KEY:
        logger.error("TAVILY_API_KEY is not configured.")
        return {"error": "API key not configured", "results": []}
        
    url = 'https://api.tavily.com/search'
    headers = {
        'Authorization': f'Bearer {TAVILY_API_KEY}',
        'Content-Type': 'application/json'
    }
    data = {
        'query': query,
        'max_results': max_results
    }
    
    try:
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()  # Raise exception for 4XX/5XX status codes
        return response.json()
    except requests.exceptions.RequestException as e:
        logger.error(f"Error during Tavily search: {e}")
        return {"error": str(e), "results": []}

# For backward compatibility
tavily_search = run_search

if __name__ == "__main__":
    query = input("Enter your search query: ")
    results = run_search(query)
    if results and "error" not in results:
        print("Search Results:")
        for idx, result in enumerate(results.get('results', []), 1):
            print(f"{idx}. {result.get('title')} - {result.get('url')}")
    else:
        print(f"Error: {results.get('error', 'Unknown error')}") 