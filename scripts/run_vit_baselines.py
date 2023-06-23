import subprocess
from pathlib import Path

model_name = "vit_b_16"

for seed in ["1", "2", "3"]:
    for dataset, lr, prune_lr, bs, epochs, recovery_epocs in [
        ("cifar10", "0.0125", "0.00125", "32", "20", "10"),
        ("oxfordflowers102", "0.0125", "0.00125", "32", "50", "20"),
        ("catsanddogs", "0.0125", "0.00125", "32", "50", "20"),
        # ("stanfordcars", "0.0125", "0.00125", "32", "50", "20"),
        # ("cifar100", "0.0125", "0.00125", "32", "20", "10"),
    ]:
        # fmt: off
        command = [
            "python", "main_vit.py",
            "--arch", model_name,
            "--dataset", dataset,
            "--lr", lr,
            "--prune_lr", prune_lr,
            "--batch-size", bs,
            "--init-epochs", epochs,
            "--recovery-epochs", recovery_epocs,
            "--seed", seed,
            # "--train",  # Added conditionally later in script
            "--prune",
        ]
        # fmt: on

        ckpt = f"weights/{model_name}_{dataset}_1.0_seed={seed}.pth"

        if Path(ckpt).exists():
            command.extend(["--resume-from-ckpt", ckpt])
        else:
            command.extend(["--train"])

        print(command)
        subprocess.call(command)
