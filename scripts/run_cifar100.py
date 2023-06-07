import subprocess

for seed in ["1"]:  # , "2", "3"]:
    # fmt: off
    command = [
        "python", "main_vit.py",
        "--arch", "vit_b_16",
        "--dataset", "cifar100",
        "--method-type", "weight",
        "--lr", "0.0025",
        "--batch-size", "64",
        "--init_epochs", "30",
        "--recovery_epochs", "20",
        "--seed", seed,
    ]
    # fmt: on
    subprocess.call(command)
