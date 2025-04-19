"""
Plan reviewer module.

This module handles the review of plans and suggestions for improvement.
"""

import os
import json
import logging
import re
from typing import Dict, List, Tuple, Optional, Any

# Import configurations
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import config
from core.openai_helper import openai_call

logger = logging.getLogger(__name__)

def review_plan(
    task: Dict[str, Any],
    plan: List[Dict[str, Any]],
    context: Dict[str, Any],
    search_results: Optional[str]
) -> Tuple[bool, Optional[str]]:
    """
    Review the proposed plan using a critical reviewer LLM.
    
    Args:
        task: Task information
        plan: Execution plan steps
        context: Context for the task
        search_results: Results from online search
        
    Returns:
        Tuple of (agree, comments) where agree is a boolean and comments are optional
    """
    review_prompt = f"""
Review the proposed plan and provide feedback on how to improve it, if necessary. If the plan is good, just say {{"agree": true}}. Think step by step. Reply ONLY in valid JSON format. Your response must be a valid JSON object. 
###
Task: {task}
###
Plan: {plan}
###
Context: {list(context.keys())}
###
Online findings: {search_results}
###

If you agree, reply with: {{"agree": true}}. 
If not, reply with: {{"agree": false, "comments": "<how to improve>"}} and suggest improvements.

- Do NOT include newlines in your JSON output.
- Keep the comments concise and actionable.
- Do NOT include any explanation or text outside the JSON object.
- If the plan is generally good, just say {{"agree": true}}.
"""
    try:
        review_response = openai_call(
            model=config.PLANNER_LLM_MODEL,
            messages=[
                {"role": "system", "content": "You are a critical reviewer for AI plans."},
                {"role": "user", "content": review_prompt}
            ],
            temperature=0.2,
            max_tokens=256,
            response_format={"type": "json_object"},
            trace_name=f"reviewer-plan",
            trace_metadata={
                "task": task,
                "plan": plan,
                "context_keys": list(context.keys()),
                "search_results": search_results
            }
        )
        review_content = review_response.choices[0].message.content.strip()
        
        # Parse the review response
        return parse_review_response(review_content)
    except Exception as e:
        logger.error(f"Error during plan review: {e}")
        return False, f"Review failed: {str(e)}"

def parse_review_response(review_content: str) -> Tuple[bool, Optional[str]]:
    """
    Parse the review response from the LLM.
    
    Args:
        review_content: Response content from the LLM
        
    Returns:
        Tuple of (agree, comments)
    """
    try:
        review_json = json.loads(review_content)
        reviewer_agrees = bool(review_json.get("agree"))
        review_comments = review_json.get("comments")
        return reviewer_agrees, review_comments
    except Exception as e:
        logger.warning(f"Failed to parse reviewer response: {e}, content: {review_content}")
        
        # Try to extract JSON object using regex
        review_json = extract_first_json_object(review_content)
        if review_json:
            reviewer_agrees = bool(review_json.get("agree"))
            review_comments = review_json.get("comments")
            logger.info("Recovered from malformed reviewer response by extracting JSON object")
            return reviewer_agrees, review_comments
            
        return False, "Reviewer response could not be parsed"

def extract_first_json_object(text: str) -> Optional[Dict[str, Any]]:
    """
    Extract the first valid JSON object from a string.
    
    Args:
        text: Text that may contain a JSON object
        
    Returns:
        Extracted JSON object or None if none found
    """
    if not text:
        return None
        
    # Try to find the first {...} or [...]
    obj_match = re.search(r'({.*?}|\[.*?\])', text, re.DOTALL)
    if obj_match:
        try:
            return json.loads(obj_match.group(0))
        except Exception:
            pass
            
    # Fallback: try to trim at the last closing brace or bracket
    match = re.search(r'[\}\]](?=[^\}\]]*$)', text)
    if match:
        trimmed = text[:match.end()]
        try:
            return json.loads(trimmed)
        except Exception:
            pass
            
    return None 