import os
import time
import subprocess
import logging

# --- Configurable paths ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MEMORY_DIR = os.path.join(BASE_DIR, "memory")
PID_FILE = os.path.join(MEMORY_DIR, "heartbeat.pid")
LOG_FILE = os.path.join(MEMORY_DIR, "defibrillator.log")
HEART_SCRIPT = os.path.join(BASE_DIR, "heart.py")
CHECK_INTERVAL = 60  # seconds

# --- Logging setup ---
logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger("defibrillator")

def is_heartbeat_alive():
    try:
        with open(PID_FILE, "r") as f:
            pid = int(f.read().strip())
        # Check if process exists
        os.kill(pid, 0)
        return True
    except Exception as e:
        logger.warning(f"Heartbeat not alive: {e}")
        return False

def rollback_and_restart():
    try:
        logger.info("Rolling back main branch to last good commit...")
        subprocess.run(["git", "checkout", "main"], cwd=BASE_DIR, check=True)
        subprocess.run(["git", "reset", "--hard", "origin/main"], cwd=BASE_DIR, check=True)
        logger.info("Restarting Telos heartbeat...")
        subprocess.Popen(["python3", HEART_SCRIPT], cwd=BASE_DIR)
        logger.info("Telos heartbeat restarted.")
    except Exception as e:
        logger.error(f"Failed to rollback and restart: {e}")

def main():
    logger.info("Defibrillator started. Monitoring Telos heartbeat...")
    while True:
        if not is_heartbeat_alive():
            logger.warning("Heartbeat not detected! Initiating rollback and restart.")
            rollback_and_restart()
        time.sleep(CHECK_INTERVAL)

if __name__ == "__main__":
    main() 