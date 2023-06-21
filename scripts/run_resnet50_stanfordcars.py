import subprocess
from pathlib import Path

model_name = "resnet50"
splora_init_range = str(1e-4)

# for seed in ["1", "2", "3"]:
for seed in ["1"]:
    for method_type, norm in [
        # ("taylor", True),
        # ("taylor", False),
        # ("grad", True),
        # ("grad", False),
        ("weight", True),
        # ("weight", False),
        # ("lrp", True),
    ]:
        for dataset, lr, bs, epochs, recovery_epocs in [
            # ("cifar10", str(0.01 / 256 * 64), "64", "40", "20"),
            # ("oxfordflowers102", str(0.01 / 256 * 6), "6", "100", "50"),
            # ("catsanddogs", str(0.01 / 256 * 16), "16", "100", "50"),
            ("stanfordcars", str(0.01 / 256 * 6), "6", "100", "50"),
        ]:
            for splora_rank in [None, "8", "32"]:
                # fmt: off
                command = [
                    "python", "main_resnet.py",
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
