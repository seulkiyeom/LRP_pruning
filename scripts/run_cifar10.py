import subprocess

model_name = "resnet50"
for method_type in [
    "lrp",
    "weight",
    "taylor",
    "grad",
]:
    # for norm in [True, False]:
    for norm in [True]:
        if not norm and method_type == "lrp":
            continue

        for seed in ["1", "2", "3"]:
            # fmt: off
            command = [
                "python", "main_resnet.py", "--train", "--prune",
                "--arch", "resnet50",
                "--dataset", "cifar10",
                "--method-type", method_type,
                "--lr", "0.0025",
                "--batch-size", "64",
                "--epochs", "20",
                "--recovery_epochs", "20",
                "--seed", seed,
            ]
            # fmt: on
            if norm:
                command.append("--norm")

        subprocess.call(command)
