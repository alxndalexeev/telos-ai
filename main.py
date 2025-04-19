#!/usr/bin/env python3
"""
Telos - A Superhuman Full-Stack Engineer

Main entry point for starting the Telos autonomous agent.
"""

import signal
import sys
from core.heart import heart_beat, close_down

def signal_handler(sig, frame):
    """Handle shutdown signals gracefully."""
    print("\nTelos received shutdown signal. Shutting down gracefully...")
    close_down()

if __name__ == "__main__":
    # Set up signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)  # Handle Ctrl+C
    signal.signal(signal.SIGTERM, signal_handler)  # Handle termination signal
    
    print("Starting Telos Heart... (press Ctrl+C to stop)")
    
    try:
        heart_beat()
    except KeyboardInterrupt:
        print("\nTelos stopped by user.")
        close_down()
    except Exception as e:
        import logging
        logging.critical(f"Telos encountered a critical error and stopped: {e}", exc_info=True)
        print(f"Telos encountered a critical error and stopped: {e}")
        close_down()
        sys.exit(1) 