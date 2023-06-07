import subprocess

for seed in ["1"]:  # , "2", "3"]:
    # fmt: off
    command = [
        "python", "main_vit.py",
        "--arch", "vit_b_16",
        "--dataset", "stanfordcars",
        # "--method-type", "weight",
        "--lr", "0.01", # should be "0.01",
        "--batch-size", "256",
        "--init-epochs", "20",
        "--recovery-epochs", "10",
        "--seed", seed,
        # "--limit-train-batches", "2",
        # "--limit-test-batches", "2",
    ]
    # fmt: on
    subprocess.call(command)
