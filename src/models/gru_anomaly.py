"""
AgHealth+ — GRU Anomaly Detection Model
=========================================
Detects anomalies in vital sign sequences using a GRU-based autoencoder.

Architecture (per paper Algorithm 4)
--------------------------------------
- Input: (batch, seq_len=24, input_size=5)
  Features: [heart_rate, spo2, steps, glucose, intake_score]
- GRU encoder: 2 layers, hidden=64
- Output head: single anomaly probability (sigmoid)
- Loss: BCE-with-logits on anomaly labels
- Threshold: 0.65 (tunable)
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from loguru import logger


# ──────────────────────────────────────────────────────────────────────────────
# GRU Model
# ──────────────────────────────────────────────────────────────────────────────

class GRUAnomalyModel(nn.Module):
    """
    GRU-based binary anomaly classifier for vital sign sequences.

    Input : (batch_size, seq_len, input_size)
    Output: (batch_size, 1)  — anomaly probability
    """

    def __init__(
        self,
        input_size: int = 5,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : Tensor (batch, seq_len, input_size)

        Returns
        -------
        logits : Tensor (batch, 1)
        """
        _, h_n = self.gru(x)          # h_n: (num_layers, batch, hidden)
        last_hidden = h_n[-1]          # take final layer's hidden state
        return self.head(last_hidden)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Returns probabilities in [0, 1]."""
        return torch.sigmoid(self.forward(x))


# ──────────────────────────────────────────────────────────────────────────────
# Normaliser
# ──────────────────────────────────────────────────────────────────────────────

class VitalNormaliser:
    """Z-score normalisation fitted on training data."""

    VITAL_KEYS = ["heart_rate", "spo2", "steps", "glucose", "intake_score"]

    def __init__(self):
        self.mean = np.zeros(len(self.VITAL_KEYS))
        self.std = np.ones(len(self.VITAL_KEYS))
        self._fitted = False

    def fit(self, sequences: List[List[Dict[str, float]]]) -> None:
        all_vals = []
        for seq in sequences:
            for step in seq:
                row = [step.get(k, 0.0) for k in self.VITAL_KEYS]
                all_vals.append(row)
        arr = np.array(all_vals)
        self.mean = arr.mean(axis=0)
        self.std = np.where(arr.std(axis=0) > 0, arr.std(axis=0), 1.0)
        self._fitted = True

    def transform(self, sequence: List[Dict[str, float]]) -> np.ndarray:
        arr = np.array([[s.get(k, 0.0) for k in self.VITAL_KEYS] for s in sequence])
        return (arr - self.mean) / self.std

    def fit_transform(self, sequences):
        self.fit(sequences)
        return [self.transform(seq) for seq in sequences]


# ──────────────────────────────────────────────────────────────────────────────
# Synthetic Vital Dataset
# ──────────────────────────────────────────────────────────────────────────────

class SyntheticVitalDataset(Dataset):
    """
    Generates synthetic 24-hour vital sign sequences with labelled anomalies.
    Anomaly rate matches realistic clinical prevalence (~15%).
    """

    def __init__(
        self,
        n_users: int = 500,
        seq_len: int = 24,
        anomaly_rate: float = 0.15,
        seed: int = 42,
    ):
        np.random.seed(seed)
        self.sequences = []
        self.labels = []

        for _ in range(n_users):
            is_anomaly = np.random.rand() < anomaly_rate
            seq = self._gen_sequence(seq_len, is_anomaly)
            self.sequences.append(seq)
            self.labels.append(float(is_anomaly))

    def _gen_sequence(self, seq_len: int, anomaly: bool) -> np.ndarray:
        # Normal baselines
        hr = np.random.normal(72, 5, seq_len)
        spo2 = np.random.normal(98, 1, seq_len)
        steps = np.cumsum(np.random.poisson(lam=50, size=seq_len)).astype(float)
        glucose = np.random.normal(90, 10, seq_len)
        intake = np.random.normal(0.8, 0.1, seq_len)  # adherence score

        if anomaly:
            # Inject an anomaly in the middle of the sequence
            mid = seq_len // 2
            event = np.random.choice(["tachycardia", "hypoglycaemia", "hypoxia"])
            if event == "tachycardia":
                hr[mid:mid+4] += np.random.uniform(30, 50, 4)
            elif event == "hypoglycaemia":
                glucose[mid:mid+3] -= np.random.uniform(30, 50, 3)
                glucose = np.clip(glucose, 30, None)
            elif event == "hypoxia":
                spo2[mid:mid+2] -= np.random.uniform(6, 10, 2)
                spo2 = np.clip(spo2, 80, 100)

        return np.stack([hr, spo2, steps / 1000, glucose, intake], axis=-1).astype(np.float32)

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return (
            torch.tensor(self.sequences[idx], dtype=torch.float32),
            torch.tensor([self.labels[idx]], dtype=torch.float32),
        )


# ──────────────────────────────────────────────────────────────────────────────
# Trainer
# ──────────────────────────────────────────────────────────────────────────────

class GRUTrainer:

    def __init__(self, model: GRUAnomalyModel, config: Dict, device: Optional[str] = None):
        self.model = model
        self.config = config
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        self.opt = optim.Adam(
            model.parameters(),
            lr=config.get("learning_rate", 1e-3),
        )
        self.criterion = nn.BCEWithLogitsLoss()
        self.threshold = config.get("threshold", 0.65)

    def train_epoch(self, loader: DataLoader) -> Dict[str, float]:
        self.model.train()
        total_loss = total_correct = total = 0

        for seqs, labels in loader:
            seqs = seqs.to(self.device)
            labels = labels.to(self.device)

            logits = self.model(seqs)
            loss = self.criterion(logits, labels)

            self.opt.zero_grad()
            loss.backward()
            self.opt.step()

            total_loss += loss.item() * seqs.size(0)
            preds = (torch.sigmoid(logits) >= self.threshold).float()
            total_correct += (preds == labels).sum().item()
            total += seqs.size(0)

        return {"loss": total_loss / total, "accuracy": total_correct / total}

    @torch.no_grad()
    def evaluate(self, loader: DataLoader) -> Dict[str, float]:
        self.model.eval()
        tp = fp = tn = fn = 0

        for seqs, labels in loader:
            seqs = seqs.to(self.device)
            labels = labels.to(self.device)
            probs = self.model.predict_proba(seqs)
            preds = (probs >= self.threshold).float()

            tp += ((preds == 1) & (labels == 1)).sum().item()
            fp += ((preds == 1) & (labels == 0)).sum().item()
            tn += ((preds == 0) & (labels == 0)).sum().item()
            fn += ((preds == 0) & (labels == 1)).sum().item()

        precision = tp / (tp + fp + 1e-9)
        recall = tp / (tp + fn + 1e-9)
        f1 = 2 * precision * recall / (precision + recall + 1e-9)
        accuracy = (tp + tn) / (tp + fp + tn + fn + 1e-9)
        return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), path)
        logger.info("GRUTrainer | model saved to {}", path)
