import os
import subprocess
import sys

# List of model scripts to run
scripts = [
    "model_svm.py",
    "model_rf.py",
    "model_dt.py",
    "model_lr.py",
    "model_mlp.py",
    "model_xgboost.py",
    "model_lightgbm.py",
    "model_catboost.py",
    "model_gb.py",
    "model_svr_linear.py",
    "model_gru.py"
]

print("Starting execution of all models...\n")

for script in scripts:
    print(f"================ Running {script} ================")
    try:
        # Run the script and wait for it to finish
        subprocess.check_call([sys.executable, script])
    except subprocess.CalledProcessError as e:
        print(f"Error running {script}: {e}")
    print("\n")

print("All models executed.")
