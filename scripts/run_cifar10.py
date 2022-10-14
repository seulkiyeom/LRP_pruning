#  Benchmark the inference of all models

# import os
import subprocess

# from pathlib import Path

# from dotenv import find_dotenv, load_dotenv

# load_dotenv(find_dotenv())

# ROOT_PATH = Path(os.getenv("ROOT_PATH", default=""))
# LOGS_PATH = Path(os.getenv("LOGS_PATH", default="logs"))
# DATASETS_PATH = Path(os.getenv("DATASETS_PATH", default="datasets"))

# DEVICE = "RTX2080Ti"
# NUM_RUNS = "100"
# BATCH_SIZE = {"CPU": 1, "RTX2080Ti": 128}[DEVICE]
# GPUS = "0" if DEVICE == "CPU" else "1"

# Regular models
model_name = "resnet50"
for method_type in [
    # "lrp",
    # "weight",
    # "taylor",
    "grad",
]:
    for norm in [
        True,
        # False,
    ]:
        if not norm and method_type == "lrp":
            continue
        # fmt: off
        command = [
            "python", "main.py", "--train", "--prune",
            "--arch", "resnet50",
            "--method-type", method_type,
            "--lr", "0.0025",
            "--batch-size", "64",
            "--epochs", "20",
            "--recovery_epochs", "20",
            # "--epochs", "2",
            # "--recovery_epochs", "2",
            # "--limit-train-batches", "5",
            # "--limit-test-batches", "5",
        ]
        # fmt: on
        if norm:
            command.append("--norm")

        subprocess.call(command)
