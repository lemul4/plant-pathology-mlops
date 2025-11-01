# src/models/run_experiments.py
import yaml
import itertools
import subprocess
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--single_model", type=str, default=None, help="If specified, run only this model")
args = parser.parse_args()

with open("params.yaml") as f:
    params = yaml.safe_load(f)

train_params = params["train"]

if args.single_model:
    experiments = [(args.single_model, True), (args.single_model, False)]
else:
    models = train_params["models"]
    use_augs = train_params["use_augmentation"]
    experiments = list(itertools.product(models, use_augs))

for model_name, aug in experiments:
    data_dir = "data_augmented" if aug else "data_processed"
    output_dir = f"models/{model_name}_{'aug' if aug else 'noaug'}"
    os.makedirs(output_dir, exist_ok=True)

    cmd = [
        "python", "src/models/train.py",
        "--data_dir", data_dir,
        "--model_name", model_name,
        "--epochs", str(train_params["epochs"]),
        "--batch_size", str(train_params["batch_size"]),
        "--lr", str(train_params["lr"]),
        "--img_size", str(train_params["img_size"]),
        "--random_seed", str(train_params["random_seed"]),
        "--output_dir", output_dir
    ]

    print(f"\nRunning experiment: model={model_name}, augmentation={aug}")
    subprocess.run(cmd, check=True)
