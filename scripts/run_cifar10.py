import subprocess

model_name = "resnet50"
for method_type in [
    "lrp",
    "weight",
    "taylor",
    "grad",
]:
    for norm in [
        True,
        False,
    ]:
        if not norm and method_type == "lrp":
            continue
        # fmt: off
        command = [
            "python", "main.py", "--train", "--prune",
            "--arch", "resnet50",
            "--dataset", "cifar10",
            "--method-type", method_type,
            "--lr", "0.0025",
            "--batch-size", "64",
            "--epochs", "20",
            "--recovery_epochs", "20",
        ]
        # fmt: on
        if norm:
            command.append("--norm")

        subprocess.call(command)