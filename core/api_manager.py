"""
API Rate Limiting and Management for Telos.

This module provides functionality for managing API rate limits
to prevent quota exhaustion and ensure sustainable operation.
"""

import time
import logging

logger = logging.getLogger(__name__)

class ApiRateLimiter:
    """Manage API rate limits for OpenAI, Tavily, etc."""
    def __init__(self):
        self.api_calls = {
            "openai": {"count": 0, "last_reset": time.time(), "limit": 200},
            "tavily": {"count": 0, "last_reset": time.time(), "limit": 50}
        }
        
    def can_make_call(self, api_name):
        """Check if we can make another API call."""
        now = time.time()
        api = self.api_calls.get(api_name.lower())
        
        if not api:
            return True
            
        # Reset counter after an hour
        if now - api["last_reset"] > 3600:
            api["count"] = 0
            api["last_reset"] = now
            
        return api["count"] < api["limit"]
        
    def record_call(self, api_name):
        """Record that an API call was made."""
        api = self.api_calls.get(api_name.lower())
        if api:
            api["count"] += 1

# Create a singleton instance
rate_limiter = ApiRateLimiter() 