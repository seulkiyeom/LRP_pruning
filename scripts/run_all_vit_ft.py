import subprocess
from pathlib import Path

model_name = "vit_b_16"

base_lr = 0.0125
base_bs = 32
bs = 16


# fmt: off
for seed in ["1"]:
    for dataset, lr, prune_lr, bs, epochs, recovery_epocs in [
        ("cifar100", str(base_lr * bs / base_bs), str(base_lr * bs / base_bs / 10), str(bs), "20", "10"),
        ("oxfordflowers102", str(base_lr * bs / base_bs), str(base_lr * bs / base_bs / 3.0), str(bs), "50", "20"),
        ("catsanddogs", str(base_lr * bs / base_bs), str(base_lr * bs / base_bs / 3.0), str(bs), "50", "20"),
        ("stanfordcars", str(base_lr * bs / base_bs), str(base_lr * bs / base_bs / 3.0), str(bs), "50", "20"),
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

        ckpt = f"weights/{model_name}_{dataset}_1.0_seed={seed}.pth"

        if Path(ckpt).exists():
            command.extend(["--resume-from-ckpt", ckpt])
        else:
            command.extend(["--train"])

        print(command)
        subprocess.call(command)

# fmt: on
