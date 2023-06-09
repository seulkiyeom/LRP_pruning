import subprocess

for seed in ["2", "3"]:
    # fmt: off
    command = [
        "python", "main_vit.py",
        "--arch", "vit_b_16",
        "--dataset", "cifar100",
        "--lr", "0.01",
        "--batch-size", "256",
        "--init-epochs", "20",
        "--recovery-epochs", "10",
        "--seed", seed,
        "--train",
        "--prune",
    ]
    # fmt: on
    subprocess.call(command)
