#!/usr/bin/env python3
"""
Telos - A Superhuman Full-Stack Engineer

Main entry point for starting the Telos autonomous agent.
"""

from core.heart import heart_beat

if __name__ == "__main__":
    print("Starting Telos... (press Ctrl+C to stop)")
    try:
        heart_beat()
    except KeyboardInterrupt:
        print("\nTelos stopped by user.")
    except Exception as e:
        import logging
        logging.critical(f"Telos encountered a critical error and stopped: {e}", exc_info=True)
        print(f"Telos encountered a critical error and stopped: {e}")
        import sys
        sys.exit(1) 