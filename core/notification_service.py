"""
Notification service for Telos AI.
Handles sending notifications to various channels (Telegram, etc.)
"""

import logging
import sys
import os
import importlib.util
from datetime import datetime
import traceback
import asyncio
from typing import Optional

import config
from telegram import Bot
from telegram.constants import ParseMode
from telegram.error import TelegramError

logger = logging.getLogger(__name__)

# Initialize notification channels
telegram_bot: Optional[Bot] = None
if config.TELEGRAM_NOTIFICATIONS_ENABLED and config.TELEGRAM_API_KEY and config.TELEGRAM_CHAT_ID:
    try:
        telegram_bot = Bot(token=config.TELEGRAM_API_KEY)
        logger.info("Telegram notification service initialized")
    except Exception as e:
        logger.error(f"Failed to initialize Telegram notification service: {e}")
        telegram_bot = None
else:
    if config.TELEGRAM_NOTIFICATIONS_ENABLED:
        logger.warning("Telegram notifications are enabled but API key or chat ID is missing")

def format_message(title, body=None, level="info", include_timestamp=True):
    """Format a message with timestamp, title, and optional body."""
    icon_map = {
        "info": "ℹ️",
        "warning": "⚠️",
        "error": "🚨",
        "success": "✅",
        "decision": "🤔",
        "action": "🔄",
        "completion": "🏁"
    }
    
    icon = icon_map.get(level.lower(), "🔹")
    timestamp = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] " if include_timestamp else ""
    
    # Use markdown formatting for Telegram
    message = f"{icon} {timestamp}*{title}*"
    if body:
        message += f"\n{body}"
    
    return message

async def send_telegram_message(chat_id, text, parse_mode=ParseMode.MARKDOWN):
    """Send a message using Telegram Bot API asynchronously."""
    if not telegram_bot:
        logger.debug("Telegram bot not initialized, skipping message")
        return False
    
    try:
        await telegram_bot.send_message(
            chat_id=chat_id,
            text=text,
            parse_mode=parse_mode
        )
        return True
    except TelegramError as e:
        logger.error(f"Telegram error: {e}")
        return False

def send_notification(title, body=None, level="info", importance="normal"):
    """
    Send a notification to all configured channels based on importance level.
    
    Args:
        title: The notification title/headline
        body: Optional detailed message
        level: Type of message (info, warning, error, success, decision, action, completion)
        importance: Importance level (minimal, normal, important)
    
    Returns:
        Success status (bool)
    """
    if not should_send_by_importance(importance):
        logger.debug(f"Skipping {importance} notification due to configuration level: {title}")
        return False
    
    success = True
    
    # Send to Telegram if enabled
    if telegram_bot:
        try:
            formatted_message = format_message(title, body, level)
            # Run the async function in a new event loop
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            result = loop.run_until_complete(
                send_telegram_message(
                    config.TELEGRAM_CHAT_ID, 
                    formatted_message
                )
            )
            loop.close()
            if result:
                logger.debug(f"Telegram notification sent: {title}")
            else:
                logger.warning(f"Failed to send Telegram notification: {title}")
                success = False
        except Exception as e:
            logger.error(f"Failed to send Telegram notification: {e}")
            success = False
    
    # Additional notification channels can be added here in the future
    
    return success

def should_send_by_importance(importance):
    """Determine if notification should be sent based on importance and configured level."""
    level_hierarchy = {
        "minimal": ["important"],
        "normal": ["important", "normal"],
        "all": ["important", "normal", "minimal"]
    }
    
    configured_level = config.TELEGRAM_NOTIFICATION_LEVEL.lower()
    if configured_level not in level_hierarchy:
        configured_level = "important"  # Default to important only
        
    return importance in level_hierarchy.get(configured_level, ["important"])

def notify_startup():
    """Send a notification when Telos starts up."""
    send_notification(
        "Telos AI is now active",
        f"System started successfully on {datetime.now().strftime('%Y-%m-%d')}",
        level="info",
        importance="normal"
    )

def notify_shutdown():
    """Send a notification when Telos shuts down."""
    send_notification(
        "Telos AI is shutting down",
        "System is going offline",
        level="info",
        importance="normal"
    )

def notify_decision(decision, details=None):
    """Send a notification about a decision made by Telos."""
    send_notification(
        f"Decision: {decision}",
        details,
        level="decision",
        importance="important"
    )

def notify_action(action, details=None):
    """Send a notification about an action taken by Telos."""
    send_notification(
        f"Action: {action}",
        details,
        level="action",
        importance="normal"
    )

def notify_task_started(task):
    """Send a notification when a task is started."""
    send_notification(
        f"Started Task: {task.get('task', 'Unknown')}",
        task.get('details', 'No details'),
        level="info",
        importance="normal"
    )

def notify_task_completed(task, success=True):
    """Send a notification when a task is completed."""
    status = "completed successfully" if success else "failed"
    send_notification(
        f"Task {status}: {task.get('task', 'Unknown')}",
        task.get('details', 'No details'),
        level="success" if success else "error",
        importance="normal"
    )

def notify_error(error_message, details=None):
    """Send a notification about an error encountered by Telos."""
    send_notification(
        f"Error: {error_message}",
        details,
        level="error",
        importance="important"
    )

def notify_code_generation(filename, purpose):
    """Send a notification about code generation."""
    send_notification(
        f"Generated code: {filename}",
        f"Purpose: {purpose}",
        level="action",
        importance="normal"
    )

def notify_code_application(target_file, source_file):
    """Send a notification about code being applied to the system."""
    send_notification(
        f"Applied code to: {target_file}",
        f"Source: {source_file}",
        level="action",
        importance="important"
    ) 