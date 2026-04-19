import time
import subprocess

while True:
    print("Running retraining...")

    try:
        subprocess.run(["python", "retrain.py"], check=True)
        print("Retraining completed successfully")
    except subprocess.CalledProcessError as e:
        print("Error during retraining:", e)

    print("⏳ Waiting for next run...\n")
    time.sleep(60)   # runs every 60 seconds