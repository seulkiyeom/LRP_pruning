import subprocess

model_name = "resnet50"
for method_type in [
    "lrp",
    # "weight",
    # "taylor",
    # "grad",
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
            "--dataset", "oxfordflowers102",
            "--method-type", method_type,
            "--lr", "0.0005",
            "--batch-size", "12",
            "--epochs", "100",
            "--recovery_epochs", "50",
        ]
        # fmt: on
        if norm:
            command.append("--norm")

        subprocess.call(command)
