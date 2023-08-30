import subprocess
from pathlib import Path

model_name = "efficientnet_v2_m"
adapter_init_range = str(1e-4)
bs = 8
lr_base = 0.256 / 4096

for seed in ["1", "2", "3"]:
    for method_type, norm in [
        ("taylor", True),
        ("grad", True),
        ("weight", True),
        ("lrp", True),
    ]:
        for dataset, lr, bs, epochs, recovery_epocs in [
            ("stanfordcars", str(lr_base * bs), str(bs), "100", "50"),
            ("oxfordflowers102", str(lr_base * bs), str(bs), "120", "50"),
            ("catsanddogs", str(lr_base * bs), str(bs), "120", "50"),
            ("cifar100", str(lr_base * bs), str(bs), "60", "30"),
            ("cifar10", str(lr_base * bs), str(bs), "50", "20"),
        ]:
            for adapter_config in [None, "8", "32", "sppara"]:
                # fmt: off
                if adapter_config is not None:
                    lr = str(2 * float(lr))
                
                command = [
                    "python", "main_cnn.py",
                    # "--train", # Added conditionally
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
                if adapter_config == "sppara":
                    command.extend([
                        "--sppara",
                        "--splora-init-range", adapter_init_range,
                    ])
                    ckpt = f"weights/sppara_{model_name}_{dataset}_1.0_seed={seed}.pth"
                elif adapter_config is not None:
                    command.extend([
                        "--splora",
                        "--splora-rank", adapter_config,
                        "--splora-init-range", adapter_init_range,
                    ])
                    ckpt = f"weights/splora_r{adapter_config}_{model_name}_{dataset}_1.0_seed={seed}.pth"
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
