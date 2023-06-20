import subprocess

model_name = "resnet50"

# CIFAR 10
for seed in [5]:
    for method_type, norm in [
        # ("weight", True),
        # ("weight", False),
        # ("taylor", True),
        # ("taylor", False),
        # ("grad", True),
        # ("grad", False),
        ("lrp", True),
    ]:
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
            "--seed", str(seed + 666),
        ]
        # fmt: on
        if norm:
            command.append("--norm")

        subprocess.call(command)


# OXFORD FLOWERS 102
for method_type, norm in [
    # ("weight", True),
    # ("weight", False),
    # ("taylor", True),
    # ("taylor", False),
    # ("grad", True),
    # ("grad", False),
    # ("lrp", True),
]:

    # fmt: off
    command = [
        "python", "main_resnet.py",
        # "--train",
        "--prune",
        "--arch", "resnet50",
        "--dataset", "oxfordflowers102",
        "--method-type", method_type,
        "--lr", "0.000156",
        "--batch-size", "4",
        "--epochs", "100",
        "--recovery_epochs", "50",
        "--resume-from-ckpt", "weights/resnet50_oxfordflowers102_1.0.pth",
    ]
    # fmt: on
    if norm:
        command.append("--norm")

    subprocess.call(command)


# CATS AND DOGS
for seed in range(3):
    for method_type, norm in [
        # ("weight", True),
        # ("weight", False),
        # ("taylor", True),
        # ("taylor", False),
        # ("grad", True),
        # ("grad", False),
        ("lrp", True),
    ]:

        # fmt: off
        command = [
            "python", "main_resnet.py",
            # "--train",
            "--prune",
            "--arch", "resnet50",
            "--dataset", "catsanddogs",
            "--method-type", method_type,
            "--lr", "0.000156",
            "--batch-size", "4",
            "--epochs", "100",
            "--recovery_epochs", "30",
            "--resume-from-ckpt", "weights/resnet50_catsanddogs_1.0.pth",
            "--seed", str(seed),
        ]
        # fmt: on
        if norm:
            command.append("--norm")

        subprocess.call(command)
