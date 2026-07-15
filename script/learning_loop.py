import subprocess
import time
import logging
import sys
from datetime import datetime


PYTHON_EXEC = sys.executable

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler("learning_loop.log"), logging.StreamHandler()]
)

logger = logging.getLogger(__name__)

print("=== AI Learning Loop (24h) ===")

SCRIPTS = [
    "train_price_model.py",
    "train_conversion_model.py",
    "train_lead_model.py",
    "train_deal_model.py",          
    "train_sales_forecast.py",   
    "evaluate_model.py"   
]

while True:
    logger.info("Starting retraining cycle -------------------")
    
    for script in SCRIPTS:
        try:
            logger.info(f"Running {script} with {PYTHON_EXEC} ...")
            result = subprocess.run(
                [PYTHON_EXEC, script],
                capture_output=True,
                text=True,
                timeout=1800
            )
            if result.returncode == 0:
                logger.info(f"{script} → OK\n{result.stdout.strip()}")
            else:
                logger.error(f"{script} failed (code {result.returncode}):\n{result.stderr}")
        except subprocess.TimeoutExpired:
            logger.error(f"{script} timed out after 30 minutes")
        except Exception as e:
            logger.exception(f"Exception while running {script}: {e}")
    
    logger.info("Cycle finished. Sleeping 24 hours...")
    time.sleep(86400)  