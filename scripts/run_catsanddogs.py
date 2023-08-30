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
                "python", "main_cnn.py",
                "--train",
                "--prune",
                "--arch", "resnet50",
                "--dataset", "catsanddogs",
                "--method-type", method_type,
                "--lr", "0.000156",
                "--batch-size", "4",
                "--epochs", "100",
                "--recovery_epochs", "50",
                "--seed", seed,
                # "--resume-from-ckpt", "weights/resnet50_catsanddogs_1.0.pth",
            ]
            # fmt: on
            if norm:
                command.append("--norm")

        subprocess.call(command)
