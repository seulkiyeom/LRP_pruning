import subprocess

for seed in ["1"]:  # ["1", "2", "3"]:
    # fmt: off
    command = [
        "python", "main_vit.py",
        "--arch", "vit_b_16",
        "--dataset", "stanfordcars",
        "--lr", "0.0125",  #"0.10",
        "--prune_lr", "0.0125",  #"0.01",
        "--batch-size", "32",  #"256",
        "--init-epochs", "50",
        "--recovery-epochs", "20",
        "--seed", seed,
        "--train",
        "--prune",
        "--global-pruning",
    ]
    # fmt: on
    subprocess.call(command)
