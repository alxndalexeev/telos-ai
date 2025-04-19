import os, logging
from langfuse.openai import openai
import config # Assuming config.py is at the project root
from core.api_manager import rate_limiter

logger = logging.getLogger(__name__)

# Load OpenAI API Key once
openai.api_key = os.getenv("OPENAI_API_KEY")
if not openai.api_key:
    logger.warning("OPENAI_API_KEY environment variable not found. OpenAI calls will fail.")

def openai_call(model: str, messages, trace_name=None, trace_metadata=None, **kwargs):
    """
    Wrapper for OpenAI chat completion calls that handles API key check and rate limiting.
    Raises ValueError if API key is not set or RuntimeError if rate limit is exceeded.
    Optionally accepts trace_name and trace_metadata for improved Langfuse trace tracking.
    """
    if not openai.api_key:
        raise ValueError("OpenAI API key is not configured.")

    if not rate_limiter.can_make_call("openai"):
        logger.warning("OpenAI API rate limit reached. Call aborted.")
        raise RuntimeError("OpenAI quota exceeded")

    # Inject Langfuse trace name and metadata if provided
    if trace_name is not None:
        kwargs["name"] = trace_name
    if trace_metadata is not None:
        kwargs["metadata"] = trace_metadata

    try:
        response = openai.chat.completions.create(
            model=model,
            messages=messages,
            **kwargs,
        )
        rate_limiter.record_call("openai")
        logger.debug(f"OpenAI call successful for model {model}. Usage: {response.usage}")
        return response
    except openai.APIError as e:
        logger.error(f"OpenAI API error during call: {e}")
        raise # Re-raise the original error after logging
    except openai.RateLimitError as e:
        # This might be redundant if rate_limiter check works perfectly, but good as a backup
        logger.error(f"OpenAI reported rate limit error: {e}")
        # We might have already recorded the call attempt, ensure counter is correct if needed
        # Or potentially adjust rate_limiter state if possible
        raise # Re-raise
    except Exception as e:
        logger.error(f"Unexpected error during OpenAI call: {e}")
        raise 