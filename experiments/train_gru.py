"""
AgHealth+ — GRU Anomaly Detector Training
===========================================
Trains the GRU anomaly detection model on synthetic vital sign sequences.

Usage:
    python experiments/train_gru.py [--epochs 30] [--n_users 500]
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from torch.utils.data import DataLoader, random_split

from src.models import GRUAnomalyModel, GRUTrainer, SyntheticVitalDataset
from src.utils import set_global_seed, save_json, load_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=5)  # 30 in production
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--n_users", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="results")
    args = parser.parse_args()

    set_global_seed(args.seed)
    cfg = load_config("model_configs").get("gru_anomaly", {})

    print("=" * 50)
    print(" GRU Anomaly Detector Training")
    print("=" * 50)

    dataset = SyntheticVitalDataset(n_users=args.n_users, seed=args.seed)
    n_val = max(30, int(len(dataset) * 0.2))
    train_ds, val_ds = random_split(dataset, [len(dataset) - n_val, n_val])
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)

    model = GRUAnomalyModel(
        input_size=cfg.get("input_size", 5),
        hidden_size=cfg.get("hidden_size", 64),
        num_layers=cfg.get("num_layers", 2),
        dropout=cfg.get("dropout", 0.2),
    )
    trainer = GRUTrainer(model, cfg)

    history = []
    for epoch in range(1, args.epochs + 1):
        train_m = trainer.train_epoch(train_loader)
        val_m = trainer.evaluate(val_loader)
        row = {"epoch": epoch, **{f"train_{k}": v for k, v in train_m.items()},
               **{f"val_{k}": v for k, v in val_m.items()}}
        history.append(row)
        print(f"  Epoch {epoch:02d} | "
              f"train_loss={train_m['loss']:.4f} | "
              f"val_f1={val_m['f1']:.3f} | "
              f"val_acc={val_m['accuracy']:.3f}")

    Path(f"{args.output_dir}/logs").mkdir(parents=True, exist_ok=True)
    trainer.save(f"{args.output_dir}/gru_anomaly.pt")
    save_json(history, f"{args.output_dir}/logs/gru_training_history.json")
    print(f"\nModel saved. Final F1: {history[-1]['val_f1']:.3f}")


if __name__ == "__main__":
    main()
