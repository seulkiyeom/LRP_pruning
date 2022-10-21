import subprocess

model_name = "resnet50"
splora_init_range = str(1e-4)

for method_type, norm in [
    ("taylor", True),
    ("taylor", False),
    ("grad", True),
    ("grad", False),
    ("weight", True),
    ("weight", False),
    ("lrp", True),
]:
    for dataset, lr, bs, epochs, recovery_epocs in [
        ("cifar10", "0.01", "256", "30", "20"),
        ("oxfordflowers102", str(0.01 / 256 * 4), "4", "100", "50"),
        ("catsanddogs", str(0.01 / 256 * 4), "4", "100", "50"),
    ]:
        for splora_rank in ["8", "32"]:
            # fmt: off
            command = [
                "python", "main.py",
                "--train",
                "--epochs", epochs,
                "--prune",
                "--recovery_epochs", recovery_epocs,
                "--arch", model_name,
                "--dataset", dataset,
                "--method-type", method_type,
                "--lr", lr,
                "--batch-size", bs,
                "--splora",
                "--splora-rank", splora_rank,
                "--splora-init-range", splora_init_range,
            ]
            # fmt: on
            if norm:
                command.append("--norm")

            subprocess.call(command)
