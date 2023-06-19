import subprocess

for seed in ["1"]:  # , "2", "3"]:
    # fmt: off
    command = [
        "python", "main_vit.py",
        "--arch", "vit_b_16",
        "--splora",
        "--splora-rank", "32",
        "--dataset", "stanfordcars",
        "--lr", "0.025", #"0.10",
        "--prune_lr", "0.0025", #"0.01",
        "--batch-size", "64", #"256",
        "--init-epochs", "50",
        "--recovery-epochs", "20",
        "--seed", seed,
        "--train",
        "--prune",
    ]
    # fmt: on
    subprocess.call(command)
