import subprocess

base_lr = 1.0
base_bs = 256
bs = 32

for seed in ["1"]:  # , "2", "3"]:
    # fmt: off
    command = [
        "python", "main_vit.py",
        "--arch", "vit_b_16",
        "--splora",
        "--splora-rank", "32",
        "--dataset", "stanfordcars",
        "--lr", str(base_lr * bs / base_bs),
        "--prune_lr", str(base_lr * bs / base_bs),
        "--batch-size", str(bs),
        "--init-epochs", "50",
        "--recovery-epochs", "20",
        "--seed", seed,
        "--train",
        "--prune",
        # "--splora-init-range", "0",
    ]
    # fmt: on
    subprocess.call(command)
