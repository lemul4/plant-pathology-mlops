import argparse
import os
import random
import numpy as np
import json
from collections import Counter
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report
)
import torch
import torch.nn as nn
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader, WeightedRandomSampler
import mlflow
import mlflow.pytorch
from mlflow.models import infer_signature


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def replace_classifier(model, model_name, num_classes):
    """Корректно заменяет последний слой под количество классов"""
    if "resnet" in model_name or "resnext" in model_name or "wide_resnet" in model_name:
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif "alexnet" in model_name:
        model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)
    elif "vgg" in model_name:
        model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)
    elif "efficientnet" in model_name:
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    elif "mobilenet_v3" in model_name:
        model.classifier[3] = nn.Linear(model.classifier[3].in_features, num_classes)
    else:
        raise ValueError(f"Неизвестная архитектура: {model_name}")
    return model


def main(data_dir, model_name, epochs, batch_size, lr, img_size, random_seed, output_dir):
    tracking_uri = "http://localhost:5000"
    experiment_name = "Plant-Pathology-Training2"
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    set_seed(random_seed)
    os.makedirs(output_dir, exist_ok=True)

    mlflow.pytorch.autolog(log_models=False)

    with mlflow.start_run(run_name=model_name) as run:
        mlflow.log_params({
            "model_name": model_name,
            "epochs": epochs,
            "batch_size": batch_size,
            "lr": lr,
            "img_size": img_size,
            "random_seed": random_seed
        })

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        transform_train = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])
        transform_val = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])

        train_dataset = datasets.ImageFolder(os.path.join(data_dir, 'train'), transform_train)
        val_dataset = datasets.ImageFolder(os.path.join(data_dir, 'val'), transform_val)

        labels = [label for _, label in train_dataset.samples]
        class_counts = Counter(labels)
        weights = [1.0 / class_counts[label] for label in labels]
        sampler = WeightedRandomSampler(weights, len(weights))

        train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler, num_workers=6)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=6)

        model = getattr(models, model_name)(pretrained=True)
        model = replace_classifier(model, model_name, len(train_dataset.classes))
        model = model.to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        best_val_acc = 0
        best_model_path = os.path.join(output_dir, 'best_model.pth')

        for epoch in range(epochs):
            model.train()
            epoch_loss = 0.0
            train_all_preds, train_all_labels = [], []
            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False)

            for images, labels in progress_bar:
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

                preds = outputs.argmax(dim=1)
                train_all_preds.extend(preds.cpu().numpy())
                train_all_labels.extend(labels.cpu().numpy())
                progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})

            avg_train_loss = epoch_loss / len(train_loader)
            train_acc = accuracy_score(train_all_labels, train_all_preds)

            model.eval()
            val_loss = 0.0
            all_preds, all_labels = [], []

            with torch.no_grad():
                for images, labels in val_loader:
                    images, labels = images.to(device), labels.to(device)
                    outputs = model(images)
                    preds = outputs.argmax(dim=1)
                    all_preds.extend(preds.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())
                    val_loss += criterion(outputs, labels).item()

            avg_val_loss = val_loss / len(val_loader)

            val_acc = accuracy_score(all_labels, all_preds)

            precision, recall, f1, support = precision_recall_fscore_support(
                all_labels, all_preds, average=None, zero_division=0
            )

            macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
                all_labels, all_preds, average="macro", zero_division=0
            )

            mlflow.log_metrics({
                "train_loss": avg_train_loss,
                "val_loss": avg_val_loss,
                "train_acc": train_acc,
                "val_acc": val_acc,
                "macro_precision": macro_p,
                "macro_recall": macro_r,
                "macro_f1": macro_f1,
            }, step=epoch + 1)

            for i, cls in enumerate(train_dataset.classes):
                mlflow.log_metrics({
                    f"{cls}_precision": precision[i],
                    f"{cls}_recall": recall[i],
                    f"{cls}_f1": f1[i],
                    f"{cls}_support": support[i]
                }, step=epoch + 1)

            cm = confusion_matrix(all_labels, all_preds)
            fig, ax = plt.subplots(figsize=(6, 5))
            sns.heatmap(cm, annot=True, fmt="d",
                        cmap="Blues", xticklabels=train_dataset.classes,
                        yticklabels=train_dataset.classes)
            plt.xlabel("Predicted")
            plt.ylabel("True")
            plt.title(f"Confusion Matrix (Epoch {epoch+1})")
            cm_path = os.path.join(output_dir, f"confusion_matrix_epoch{epoch+1}.png")
            plt.savefig(cm_path, bbox_inches='tight')
            plt.close(fig)
            mlflow.log_artifact(cm_path)

            report = classification_report(all_labels, all_preds, target_names=train_dataset.classes)
            report_path = os.path.join(output_dir, f"classification_report_epoch{epoch+1}.txt")
            with open(report_path, 'w') as f:
                f.write(report)
            mlflow.log_artifact(report_path)

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(model.state_dict(), best_model_path)
                input_example = torch.randn(1, 3, img_size, img_size).to(device)
                signature = infer_signature(input_example.cpu().numpy(), model(input_example).cpu().detach().numpy())
                mlflow.pytorch.log_model(model, name="best_model", signature=signature, input_example=input_example.cpu().numpy())
                mlflow.log_artifact(best_model_path)

            tqdm.write(f"Epoch {epoch+1}/{epochs} | train_loss={avg_train_loss:.4f} | val_loss={avg_val_loss:.4f} | val_acc={val_acc:.4f}")

        metrics = {
            "best_val_acc": best_val_acc,
            "macro_f1": macro_f1
        }
        metrics_path = os.path.join(output_dir, 'train_metrics.json')
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=4)
        mlflow.log_artifact(metrics_path)
        mlflow.log_artifacts(output_dir)

        print(f"\n✅ Training complete. Best val_acc={best_val_acc:.4f}")
        print(f"View run {model_name} at: {tracking_uri}/#/experiments/1/runs/{run.info.run_id}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--model_name", type=str, default="resnet50")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--random_seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="models")
    args = parser.parse_args()

    main(
        args.data_dir,
        args.model_name,
        args.epochs,
        args.batch_size,
        args.lr,
        args.img_size,
        args.random_seed,
        args.output_dir
    )
