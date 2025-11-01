import argparse
import os
import time
import json
import torch
import torch.nn as nn
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns

def evaluate(model_path, data_dir, batch_size, img_size, output_dir, model_name="resnet18"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(output_dir, exist_ok=True)

    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
    val_dataset = datasets.ImageFolder(os.path.join(data_dir, 'val'), transform)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    model = getattr(models, model_name)(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, len(val_dataset.classes))
    model.load_state_dict(torch.load(os.path.join(model_path, 'best_model.pth'), map_location=device))
    model = model.to(device)
    model.eval()

    criterion = nn.CrossEntropyLoss()

    all_preds, all_labels = [], []
    total_loss = 0.0
    total_images = 0
    start_time = time.time()

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            total_images += images.size(0)

    end_time = time.time()
    eval_time = end_time - start_time
    speed_per_image = total_images / eval_time

    avg_loss = total_loss / total_images
    acc = accuracy_score(all_labels, all_preds)

    precision, recall, f1, support = precision_recall_fscore_support(
        all_labels, all_preds, average=None, zero_division=0
    )
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average="macro", zero_division=0
    )

    cm = confusion_matrix(all_labels, all_preds)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d",
                cmap="Blues", xticklabels=val_dataset.classes,
                yticklabels=val_dataset.classes)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    cm_path = os.path.join(output_dir, "confusion_matrix.png")
    plt.savefig(cm_path, bbox_inches='tight')
    plt.close(fig)

    report = classification_report(all_labels, all_preds, target_names=val_dataset.classes)
    report_path = os.path.join(output_dir, "classification_report.txt")
    with open(report_path, "w") as f:
        f.write(report)
    metrics = {
        "eval_loss": avg_loss,
        "eval_acc": acc,
        "macro_precision": macro_p,
        "macro_recall": macro_r,
        "macro_f1": macro_f1,
        "speed_images_per_sec": speed_per_image
    }
    for i, cls in enumerate(val_dataset.classes):
        metrics[f"{cls}_precision"] = precision[i]
        metrics[f"{cls}_recall"] = recall[i]
        metrics[f"{cls}_f1"] = f1[i]
        metrics[f"{cls}_support"] = support[i]
    metrics_path = os.path.join(output_dir, "eval_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)
    print(f"\n✅ Evaluation complete. Accuracy={acc:.4f}, Speed={speed_per_image:.2f} img/sec")
    print(f"Reports saved to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, required=True)
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="reports")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--model_name", type=str, default="resnet18")
    args = parser.parse_args()

    evaluate(
        model_path=args.model_dir,
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        img_size=args.img_size,
        output_dir=args.output_dir,
        model_name=args.model_name
    )
