"""
Centralized Logging Module for Telos.

This module provides logging functionality for actions and thoughts
that Telos takes during execution.
"""

import logging
import os
from datetime import datetime
from typing import Any

import config

# Setup logging based on config
logging.basicConfig(level=config.LOG_LEVEL, format=config.LOG_FORMAT)

def log_action(action: str, result: Any) -> None:
    """Log an action and its result to the action log."""
    now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    # Ensure result is a string and escape characters that might break the table format
    result_str = str(result).replace('|', '\\|').replace('\n', ' ')
    log_entry = f"| {now} | {action} | {result_str} |\n"
    try:
        with open(config.ACTION_LOG, 'a', encoding='utf-8') as f:
            # Add header if file is new/empty
            if f.tell() == 0:
                 header = f'''# Action Log

| Timestamp | Action | Result |
|---|---|---|
'''
                 f.write(header)
            f.write(log_entry)
    except Exception as e:
        logging.error(f"Failed to write to action log {config.ACTION_LOG}: {e}")

def log_thought(thought: str) -> None:
    """Log a thought to the thoughts log."""
    now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    log_entry = f"- [{now}] {thought}\n"
    try:
        with open(config.THOUGHTS_LOG, 'a', encoding='utf-8') as f:
            # Add header if file is new/empty
            if f.tell() == 0:
                 header = f'''# Thoughts Log

'''
                 f.write(header)
            f.write(log_entry)
    except Exception as e:
        logging.error(f"Failed to write to thoughts log {config.THOUGHTS_LOG}: {e}") 