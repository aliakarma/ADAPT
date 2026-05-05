"""
AgHealth+ — CNN Food Classifier Training
==========================================
Trains the MobileNetV2 food classifier on synthetic data.
Replace SyntheticFoodDataset with a real dataset (e.g., Food-101) for production.

Usage:
    python experiments/train_cnn.py [--epochs 20] [--batch_size 32]
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from torch.utils.data import DataLoader, random_split

from src.models import FoodClassifierCNN, CNNTrainer, SyntheticFoodDataset
from src.utils import set_global_seed, save_json, load_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=5)  # 20 in production
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--n_samples", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="results")
    args = parser.parse_args()

    set_global_seed(args.seed)
    cfg = load_config("model_configs").get("cnn_food_classifier", {})

    print("=" * 50)
    print(" CNN Food Classifier Training")
    print("=" * 50)

    # Dataset
    dataset = SyntheticFoodDataset(n_samples=args.n_samples, seed=args.seed)
    n_val = max(50, int(len(dataset) * 0.2))
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(dataset, [n_train, n_val])
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)

    # Model + Trainer
    model = FoodClassifierCNN(num_classes=10, pretrained=False)  # False for CI
    trainer = CNNTrainer(model, cfg)

    history = []
    for epoch in range(1, args.epochs + 1):
        train_metrics = trainer.train_epoch(train_loader)
        val_metrics = trainer.evaluate(val_loader)
        row = {"epoch": epoch, **{f"train_{k}": v for k, v in train_metrics.items()},
               **{f"val_{k}": v for k, v in val_metrics.items()}}
        history.append(row)
        print(f"  Epoch {epoch:02d} | "
              f"train_acc={train_metrics['accuracy']:.3f} | "
              f"val_acc={val_metrics['accuracy']:.3f} | "
              f"val_mae_cal={val_metrics.get('calories', 0):.1f}")

    # Save
    Path(f"{args.output_dir}/logs").mkdir(parents=True, exist_ok=True)
    trainer.save(f"{args.output_dir}/cnn_food_classifier.pt")
    save_json(history, f"{args.output_dir}/logs/cnn_training_history.json")
    print(f"\nModel saved. Final val accuracy: {history[-1]['val_accuracy']:.3f}")


if __name__ == "__main__":
    main()
