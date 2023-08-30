import subprocess
from pathlib import Path

model_name = "resnet18"
splora_init_range = str(1e-4)

for seed in ["1"]:
    for method_type, norm in [
        ("weight", True),
    ]:
        for dataset, lr, bs, epochs, recovery_epocs in [
            ("cifar10", str(0.01 / 256 * 64), "64", "40", "40"),
        ]:
            for splora_rank in [None, "8", "32"]:
                for pr_step in [
                    0.95,
                    0.75,
                    0.45,
                    0.15,
                    0.05,
                    0.1,
                    0.2,
                    0.25,
                    0.3,
                    0.35,
                    0.4,
                    0.5
                    0.55,
                    0.6,
                    0.65,
                    0.7,
                    0.8,
                    0.85,
                    0.9,
                ]:
                    # fmt: off
                    command = [
                        "python", "main_cnn.py",
                        # "--train",
                        "--epochs", epochs,
                        "--prune",
                        "--recovery_epochs", recovery_epocs,
                        "--arch", model_name,
                        "--dataset", dataset,
                        "--method-type", method_type,
                        "--lr", lr,
                        "--batch-size", bs,
                        "--seed", seed,
                        "--total-pr", str(min(1.0, 2 * pr_step - 0.05)),
                        "--pr-step", str(pr_step),
                    ]
                    if splora_rank is not None:
                        command.extend([
                            "--splora",
                            "--splora-rank", splora_rank,
                            "--splora-init-range", splora_init_range,
                        ])
                        ckpt = f"weights/splora_r{splora_rank}_{model_name}_{dataset}_1.0_seed={seed}.pth"
                    else:
                        ckpt = f"weights/{model_name}_{dataset}_1.0_seed={seed}.pth"
                    # fmt: on
                    if norm:
                        command.append("--norm")

                    if Path(ckpt).exists():
                        command.extend(["--resume-from-ckpt", ckpt])
                    else:
                        command.extend(["--train"])

                    print(command)
                    subprocess.call(command)
