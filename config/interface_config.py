"""
Interface configuration for Telos AI.

This module provides configuration settings for various interfaces including
CLI, Web UI, REST API, and chat interfaces.
"""

from enum import Enum
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass

class InterfaceType(Enum):
    """Enumeration of supported interface types."""
    CLI = "cli"
    WEB_UI = "web_ui"
    REST_API = "rest_api"
    CHAT = "chat"
    VOICE = "voice"
    WEBSOCKET = "websocket"
    TELEGRAM = "telegram"
    SLACK = "slack"
    DISCORD = "discord"
    EMAIL = "email"

class ThemeType(Enum):
    """Enumeration of supported theme types."""
    LIGHT = "light"
    DARK = "dark"
    SYSTEM = "system"
    CUSTOM = "custom"

class LogLevel(Enum):
    """Enumeration of log levels."""
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

# Global interface settings
GLOBAL_INTERFACE_CONFIG = {
    "default_interface": InterfaceType.CLI,
    "enable_multi_interface": True,
    "theme": ThemeType.SYSTEM,
    "language": "en",
    "debug_mode": False,
    "log_level": LogLevel.INFO,
    "auto_save_conversation": True,
    "max_conversation_history": 100,
    "enable_analytics": False,
    "enable_feedback": True,
    "enable_help_command": True,
    "enable_version_check": True,
    "enable_auto_update": False,
    "cache_timeout": 3600,  # in seconds
    "active_session_timeout": 1800,  # in seconds
    "max_parallel_sessions": 5,
}

# Command Line Interface configuration
CLI_CONFIG = {
    "enabled": True,
    "prompt_style": "telos> ",
    "show_model_info": True,
    "show_thinking": False,
    "show_timestamps": True,
    "show_token_count": False,
    "multiline_input": True,
    "syntax_highlighting": True,
    "autocomplete": True,
    "command_history_size": 1000,
    "command_history_file": ".telos_history",
    "editor": "vim",
    "editor_args": [],
    "pager": "less",
    "pager_args": ["-R"],
    "default_output_format": "text",
    "output_formats": ["text", "json", "yaml", "markdown"],
    "table_format": "grid",
    "colors": {
        "prompt": "bright_cyan",
        "input": "white",
        "system_message": "bright_black",
        "output": "bright_white",
        "error": "bright_red",
        "warning": "bright_yellow",
        "info": "bright_blue",
        "debug": "bright_magenta",
    },
}

# Web UI configuration
WEB_UI_CONFIG = {
    "enabled": True,
    "host": "localhost",
    "port": 8000,
    "debug": False,
    "open_browser": True,
    "require_login": False,
    "session_timeout": 3600,  # in seconds
    "max_upload_size": 50 * 1024 * 1024,  # 50MB
    "enable_file_upload": True,
    "enable_file_download": True,
    "enable_streaming": True,
    "enable_dark_mode": True,
    "enable_voice_input": True,
    "enable_voice_output": True,
    "max_messages_displayed": 100,
    "polling_interval": 1000,  # in milliseconds
    "theme_options": {
        "primary_color": "#1E88E5",
        "secondary_color": "#26A69A",
        "accent_color": "#FF4081",
        "background_color": "#FFFFFF",
        "text_color": "#212121",
        "font_family": "Roboto, sans-serif",
        "code_font_family": "JetBrains Mono, monospace",
    },
    "ui_components": {
        "sidebar": True,
        "settings_panel": True,
        "message_history": True,
        "model_selector": True,
        "temperature_slider": True,
        "token_counter": True,
        "save_button": True,
        "clear_button": True,
        "feedback_buttons": True,
        "file_attachments": True,
    },
}

# REST API configuration
REST_API_CONFIG = {
    "enabled": True,
    "host": "localhost",
    "port": 8080,
    "debug": False,
    "api_prefix": "/api/v1",
    "require_api_key": True,
    "rate_limit": {
        "enabled": True,
        "requests_per_minute": 60,
        "burst": 10,
    },
    "cors": {
        "enabled": True,
        "allow_origins": ["*"],
        "allow_methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization"],
    },
    "timeout": 60,  # in seconds
    "max_request_size": 10 * 1024 * 1024,  # 10MB
    "response_compression": True,
    "documentation": {
        "enabled": True,
        "title": "Telos AI API",
        "description": "REST API for interacting with Telos AI",
        "version": "1.0.0",
    },
    "endpoints": {
        "chat": "/chat",
        "memory": "/memory",
        "tools": "/tools",
        "files": "/files",
        "tasks": "/tasks",
        "users": "/users",
        "health": "/health",
    },
}

# Chat interface configuration
CHAT_CONFIG = {
    "enabled": True,
    "default_system_message": "You are Telos, a helpful AI assistant.",
    "max_history_length": 50,
    "streaming": True,
    "show_thinking": False,
    "thinking_style": "markdown",
    "default_model": "gpt-4.1",
    "default_temperature": 0.7,
    "default_max_tokens": 1024,
    "formatting": {
        "code_highlighting": True,
        "markdown_rendering": True,
        "math_rendering": True,
        "table_formatting": True,
        "linkify_urls": True,
        "emoji_support": True,
    },
    "persona_templates": {
        "default": "You are Telos, a helpful AI assistant.",
        "expert": "You are Telos, an expert AI assistant with deep knowledge in {field}.",
        "creative": "You are Telos, a creative AI assistant that specializes in generating imaginative ideas.",
        "concise": "You are Telos, an AI assistant that provides brief, to-the-point responses.",
        "academic": "You are Telos, an AI assistant with academic expertise. Respond in a scholarly style with references."
    },
    "suggestions": {
        "enabled": True,
        "count": 3,
        "based_on_history": True,
    },
    "notifications": {
        "sound": True,
        "desktop": False,
        "typing_indicator": True,
    },
}

# Voice interface configuration
VOICE_CONFIG = {
    "enabled": True,
    "tts_provider": "openai",  # Options: "openai", "eleven_labs", "google", "local"
    "stt_provider": "openai",  # Options: "openai", "whisper", "google", "local"
    "tts_voice": "alloy",  # Default voice for OpenAI
    "tts_speed": 1.0,
    "auto_detect_language": True,
    "default_language": "en-US",
    "activation_phrase": "hey telos",
    "wake_word_detection": {
        "enabled": True,
        "sensitivity": 0.5,
        "local_model_path": "models/wake_word/hey_telos.pth",
    },
    "noise_suppression": True,
    "echo_cancellation": True,
    "auto_gain_control": True,
    "vad_sensitivity": 0.5,  # Voice Activity Detection
    "input_device": "default",
    "output_device": "default",
    "timeout": 10,  # seconds of silence before stopping listening
    "response_prefix_audio": "sounds/response-start.mp3",
    "response_suffix_audio": "sounds/response-end.mp3",
}

# WebSocket configuration
WEBSOCKET_CONFIG = {
    "enabled": True,
    "host": "localhost",
    "port": 8765,
    "path": "/ws",
    "max_connections": 100,
    "max_message_size": 1 * 1024 * 1024,  # 1MB
    "ping_interval": 30,  # in seconds
    "ping_timeout": 10,  # in seconds
    "close_timeout": 10,  # in seconds
    "compression": True,
}

# Messaging platform configurations
TELEGRAM_CONFIG = {
    "enabled": False,
    "api_token": "",
    "webhook_url": "",
    "polling_interval": 2,  # in seconds
    "allowed_user_ids": [],  # Empty list means all users are allowed
    "max_message_length": 4096,
    "command_prefix": "/",
    "default_commands": {
        "start": "Start a conversation with Telos",
        "help": "Show available commands",
        "settings": "Adjust conversation settings",
        "clear": "Clear conversation history",
    },
}

SLACK_CONFIG = {
    "enabled": False,
    "bot_token": "",
    "signing_secret": "",
    "app_token": "",
    "allowed_channels": [],  # Empty list means all channels are allowed
    "default_commands": {
        "help": "Show available commands",
        "settings": "Adjust conversation settings",
        "clear": "Clear conversation history",
    },
}

DISCORD_CONFIG = {
    "enabled": False,
    "bot_token": "",
    "command_prefix": "!telos",
    "allowed_servers": [],  # Empty list means all servers are allowed
    "allowed_channels": [],  # Empty list means all channels are allowed
    "max_message_length": 2000,
    "default_commands": {
        "help": "Show available commands",
        "settings": "Adjust conversation settings",
        "clear": "Clear conversation history",
    },
}

EMAIL_CONFIG = {
    "enabled": False,
    "check_interval": 300,  # in seconds
    "smtp_server": "",
    "smtp_port": 587,
    "imap_server": "",
    "imap_port": 993,
    "username": "",
    "password": "",
    "use_tls": True,
    "allowed_senders": [],  # Empty list means all senders are allowed
    "subject_prefix": "[Telos]",
    "signature": "\n\n---\nPowered by Telos AI",
    "max_attachment_size": 10 * 1024 * 1024,  # 10MB
}

# Interface registry
INTERFACE_CONFIGS = {
    InterfaceType.CLI: CLI_CONFIG,
    InterfaceType.WEB_UI: WEB_UI_CONFIG,
    InterfaceType.REST_API: REST_API_CONFIG,
    InterfaceType.CHAT: CHAT_CONFIG,
    InterfaceType.VOICE: VOICE_CONFIG,
    InterfaceType.WEBSOCKET: WEBSOCKET_CONFIG,
    InterfaceType.TELEGRAM: TELEGRAM_CONFIG,
    InterfaceType.SLACK: SLACK_CONFIG,
    InterfaceType.DISCORD: DISCORD_CONFIG,
    InterfaceType.EMAIL: EMAIL_CONFIG,
}

def get_interface_config(interface_type: InterfaceType) -> Dict[str, Any]:
    """
    Get configuration for a specific interface type.
    
    Args:
        interface_type: The interface type to get configuration for
        
    Returns:
        Dictionary with interface configuration
    """
    if interface_type in INTERFACE_CONFIGS:
        return INTERFACE_CONFIGS[interface_type].copy()
    
    # If interface type not found, return empty config
    return {}

def get_global_config() -> Dict[str, Any]:
    """
    Get global interface configuration.
    
    Returns:
        Dictionary with global configuration
    """
    return GLOBAL_INTERFACE_CONFIG.copy()

def is_interface_enabled(interface_type: InterfaceType) -> bool:
    """
    Check if a specific interface is enabled.
    
    Args:
        interface_type: The interface type to check
        
    Returns:
        True if enabled, False otherwise
    """
    config = get_interface_config(interface_type)
    return config.get("enabled", False)

def get_theme_settings(theme: ThemeType = None) -> Dict[str, Any]:
    """
    Get theme settings.
    
    Args:
        theme: The theme type, or None for default theme
        
    Returns:
        Dictionary with theme settings
    """
    if theme is None:
        theme = GLOBAL_INTERFACE_CONFIG["theme"]
    
    if theme == ThemeType.CUSTOM:
        return WEB_UI_CONFIG["theme_options"].copy()
    
    # Predefined themes
    if theme == ThemeType.DARK:
        return {
            "primary_color": "#2196F3",
            "secondary_color": "#009688",
            "accent_color": "#FF4081",
            "background_color": "#121212",
            "text_color": "#FFFFFF",
            "font_family": "Roboto, sans-serif",
            "code_font_family": "JetBrains Mono, monospace",
        }
    
    # Light theme (default)
    return {
        "primary_color": "#1E88E5",
        "secondary_color": "#26A69A",
        "accent_color": "#FF4081",
        "background_color": "#FFFFFF",
        "text_color": "#212121",
        "font_family": "Roboto, sans-serif",
        "code_font_family": "JetBrains Mono, monospace",
    }

def get_enabled_interfaces() -> List[InterfaceType]:
    """
    Get a list of all enabled interfaces.
    
    Returns:
        List of enabled interface types
    """
    return [
        interface_type
        for interface_type in InterfaceType
        if is_interface_enabled(interface_type)
    ]

def validate_interface_config(interface_type: InterfaceType, config: Dict[str, Any]) -> List[str]:
    """
    Validate configuration for a specific interface type.
    
    Args:
        interface_type: The interface type
        config: The configuration to validate
        
    Returns:
        List of validation errors, empty if valid
    """
    errors = []
    
    # Basic validation
    if not isinstance(config, dict):
        errors.append("Configuration must be a dictionary")
        return errors
    
    # Interface-specific validation
    if interface_type == InterfaceType.CLI:
        if "prompt_style" in config and not isinstance(config["prompt_style"], str):
            errors.append("prompt_style must be a string")
        
        if "command_history_size" in config and not isinstance(config["command_history_size"], int):
            errors.append("command_history_size must be an integer")
    
    elif interface_type == InterfaceType.WEB_UI:
        if "port" in config and not isinstance(config["port"], int):
            errors.append("port must be an integer")
        
        if "max_upload_size" in config and not isinstance(config["max_upload_size"], int):
            errors.append("max_upload_size must be an integer")
    
    elif interface_type == InterfaceType.REST_API:
        if "port" in config and not isinstance(config["port"], int):
            errors.append("port must be an integer")
        
        if "api_prefix" in config and not isinstance(config["api_prefix"], str):
            errors.append("api_prefix must be a string")
    
    # Check for required fields
    required_fields = {
        InterfaceType.CLI: ["enabled", "prompt_style"],
        InterfaceType.WEB_UI: ["enabled", "host", "port"],
        InterfaceType.REST_API: ["enabled", "host", "port", "api_prefix"],
        InterfaceType.CHAT: ["enabled", "default_system_message"],
        InterfaceType.VOICE: ["enabled", "tts_provider", "stt_provider"],
        InterfaceType.WEBSOCKET: ["enabled", "host", "port", "path"],
        InterfaceType.TELEGRAM: ["enabled"],
        InterfaceType.SLACK: ["enabled"],
        InterfaceType.DISCORD: ["enabled"],
        InterfaceType.EMAIL: ["enabled"],
    }
    
    if interface_type in required_fields:
        for field in required_fields[interface_type]:
            if field not in config:
                errors.append(f"Missing required field: {field}")
    
    return errors

def get_persona_template(persona_name: str = "default") -> Optional[str]:
    """
    Get a persona template by name.
    
    Args:
        persona_name: The name of the persona template
        
    Returns:
        Template string if found, None otherwise
    """
    persona_templates = CHAT_CONFIG.get("persona_templates", {})
    return persona_templates.get(persona_name)

def update_interface_config(interface_type: InterfaceType, updates: Dict[str, Any]) -> Dict[str, Any]:
    """
    Update configuration for a specific interface type.
    
    This function does not persist changes to disk. It only updates the in-memory configuration.
    
    Args:
        interface_type: The interface type to update
        updates: Dictionary with configuration updates
        
    Returns:
        Updated configuration dictionary
    """
    if interface_type not in INTERFACE_CONFIGS:
        return {}
    
    config = INTERFACE_CONFIGS[interface_type].copy()
    config.update(updates)
    INTERFACE_CONFIGS[interface_type] = config
    
    return config 