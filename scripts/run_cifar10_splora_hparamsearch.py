import subprocess

model_name = "resnet18"

for splora_rank in [1, 2, 4, 8, 16, 32, 64]:
    for splora_init_range in [1e-6, 1e-5, 1e-4, 1e-3, 1e-2]:
        for method_type in [
            # "lrp",
            # "weight",
            "taylor",
            # "grad",
        ]:
            for norm in [
                True,  # False,
            ]:
                if not norm and method_type == "lrp":
                    continue
                # fmt: off
                command = [
                    "python", "main_cnn.py",
                    "--train",
                    "--epochs", "20",
                    "--prune",
                    "--recovery_epochs", "20",
                    "--arch", model_name,
                    "--dataset", "cifar10",
                    "--method-type", method_type,
                    "--lr", "0.01",
                    "--batch-size", "256",
                    "--splora",
                    "--splora-rank", str(splora_rank),
                    "--splora-init-range", str(splora_init_range),
                ]
                # fmt: on
                if norm:
                    command.append("--norm")

                subprocess.call(command)
