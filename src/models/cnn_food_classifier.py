"""
AgHealth+ — CNN Food Classifier (MobileNetV2)
================================================
Production-ready PyTorch implementation of the food recognition model.
Reports 99% accuracy on 10-class food dataset (paper Figure 3).

Architecture
------------
- Backbone: MobileNetV2 (pretrained on ImageNet)
- Fine-tuned last 3 layers
- Classification head: 1280 → 256 → 10
- Regression heads: 1280 → 128 → [calories, protein, fat, sugar]
  (paper Figure 4 — MAE: calories=13.9, protein=1.2, fat=1.3, sugar=1.1)
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from loguru import logger


# ──────────────────────────────────────────────────────────────────────────────
# Model
# ──────────────────────────────────────────────────────────────────────────────

class FoodClassifierCNN(nn.Module):
    """
    MobileNetV2 backbone with:
    - Softmax classification head (10 food classes)
    - Regression head for macronutrient prediction
    """

    FOOD_CLASSES = [
        "salad", "pasta", "rice_dish", "soup", "sandwich",
        "fruit_bowl", "protein_plate", "legume_dish", "dairy_product", "snack_item"
    ]
    NUTRIENT_OUTPUTS = ["calories", "protein_g", "fat_g", "sugar_g"]

    def __init__(self, num_classes: int = 10, pretrained: bool = True, dropout: float = 0.3):
        super().__init__()
        self.num_classes = num_classes

        # Backbone
        weights = models.MobileNet_V2_Weights.DEFAULT if pretrained else None
        backbone = models.mobilenet_v2(weights=weights)

        # Freeze early layers; unfreeze last 3
        for param in backbone.parameters():
            param.requires_grad = False
        for layer in list(backbone.features)[-3:]:
            for param in layer.parameters():
                param.requires_grad = True

        self.features = backbone.features
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        feature_dim = 1280  # MobileNetV2 output channels

        # Classification head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout / 2),
            nn.Linear(256, num_classes),
        )

        # Nutrient regression heads
        self.nutrient_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(feature_dim, 128),
            nn.ReLU(),
            nn.Linear(128, len(self.NUTRIENT_OUTPUTS)),
            nn.ReLU(),  # nutrients are non-negative
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        feat = self.features(x)
        feat = self.pool(feat)
        feat = feat.view(feat.size(0), -1)
        return {
            "logits": self.classifier(feat),
            "nutrients": self.nutrient_head(feat),
        }

    def predict(self, image_tensor: torch.Tensor) -> Dict[str, object]:
        """Inference helper returning class label + nutrient estimates."""
        self.eval()
        with torch.no_grad():
            out = self.forward(image_tensor.unsqueeze(0))
            probs = torch.softmax(out["logits"], dim=-1)
            conf, idx = probs.max(dim=-1)
            nutrients = out["nutrients"].squeeze(0).tolist()
        return {
            "class": self.FOOD_CLASSES[idx.item()],
            "confidence": conf.item(),
            "probabilities": {c: probs[0, i].item() for i, c in enumerate(self.FOOD_CLASSES)},
            "nutrients": {
                k: round(nutrients[i], 1)
                for i, k in enumerate(self.NUTRIENT_OUTPUTS)
            },
        }


# ──────────────────────────────────────────────────────────────────────────────
# Image transforms
# ──────────────────────────────────────────────────────────────────────────────

def get_train_transforms(img_size: int = 224) -> transforms.Compose:
    return transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def get_eval_transforms(img_size: int = 224) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize(int(img_size * 1.14)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


# ──────────────────────────────────────────────────────────────────────────────
# Trainer
# ──────────────────────────────────────────────────────────────────────────────

class CNNTrainer:
    """
    Training loop for FoodClassifierCNN.
    Trains classification + nutrient regression jointly.
    """

    def __init__(
        self,
        model: FoodClassifierCNN,
        config: Dict,
        device: Optional[str] = None,
    ):
        self.model = model
        self.config = config
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        self.opt = optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=config.get("learning_rate", 3e-4),
            weight_decay=config.get("weight_decay", 1e-4),
        )
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.opt, T_max=config.get("epochs", 20)
        )
        self.cls_criterion = nn.CrossEntropyLoss()
        self.reg_criterion = nn.L1Loss()  # MAE loss

    def train_epoch(self, loader: DataLoader) -> Dict[str, float]:
        self.model.train()
        total_cls_loss = total_reg_loss = total_correct = total = 0

        for imgs, labels, nutrients in loader:
            imgs = imgs.to(self.device)
            labels = labels.to(self.device)
            nutrients = nutrients.to(self.device).float()

            out = self.model(imgs)
            cls_loss = self.cls_criterion(out["logits"], labels)
            reg_loss = self.reg_criterion(out["nutrients"], nutrients)
            loss = cls_loss + 0.1 * reg_loss  # weighted multi-task

            self.opt.zero_grad()
            loss.backward()
            self.opt.step()

            total_cls_loss += cls_loss.item() * imgs.size(0)
            total_reg_loss += reg_loss.item() * imgs.size(0)
            preds = out["logits"].argmax(dim=-1)
            total_correct += (preds == labels).sum().item()
            total += imgs.size(0)

        self.scheduler.step()
        return {
            "cls_loss": total_cls_loss / total,
            "reg_loss": total_reg_loss / total,
            "accuracy": total_correct / total,
        }

    @torch.no_grad()
    def evaluate(self, loader: DataLoader) -> Dict[str, float]:
        self.model.eval()
        total_correct = total = 0
        mae_accum = {k: 0.0 for k in FoodClassifierCNN.NUTRIENT_OUTPUTS}

        for imgs, labels, nutrients in loader:
            imgs = imgs.to(self.device)
            labels = labels.to(self.device)
            nutrients = nutrients.to(self.device).float()

            out = self.model(imgs)
            preds = out["logits"].argmax(dim=-1)
            total_correct += (preds == labels).sum().item()
            total += imgs.size(0)

            diff = (out["nutrients"] - nutrients).abs().mean(dim=0)
            for i, k in enumerate(FoodClassifierCNN.NUTRIENT_OUTPUTS):
                mae_accum[k] += diff[i].item() * imgs.size(0)

        mae = {k: v / total for k, v in mae_accum.items()}
        return {"accuracy": total_correct / total, **mae}

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), path)
        logger.info("CNNTrainer | model saved to {}", path)

    def load(self, path: str) -> None:
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        logger.info("CNNTrainer | model loaded from {}", path)


# ──────────────────────────────────────────────────────────────────────────────
# Synthetic dataset (for reproducibility without real data)
# ──────────────────────────────────────────────────────────────────────────────

class SyntheticFoodDataset(Dataset):
    """
    Generates synthetic (random noise) food images for pipeline testing.
    Replace with a real dataset (e.g., Food-101) for production training.
    """

    NUTRIENT_RANGES = {
        "salad":         ([60, 100],  [2, 4],   [3, 6],   [2, 4]),
        "pasta":         ([180, 260], [6, 10],  [8, 14],  [1, 4]),
        "rice_dish":     ([150, 220], [3, 6],   [1, 4],   [0, 2]),
        "soup":          ([80, 140],  [4, 8],   [2, 5],   [2, 5]),
        "sandwich":      ([220, 300], [10, 15], [7, 12],  [3, 6]),
        "fruit_bowl":    ([70, 110],  [1, 2],   [0, 1],   [15, 22]),
        "protein_plate": ([200, 280], [25, 35], [8, 14],  [0, 2]),
        "legume_dish":   ([140, 200], [8, 14],  [2, 5],   [1, 3]),
        "dairy_product": ([100, 150], [6, 10],  [4, 8],   [8, 13]),
        "snack_item":    ([150, 250], [2, 5],   [8, 14],  [8, 16]),
    }

    def __init__(self, n_samples: int = 1000, img_size: int = 224, seed: int = 42):
        import random as _r
        _r.seed(seed)
        self.samples = []
        classes = list(self.NUTRIENT_RANGES.keys())
        for _ in range(n_samples):
            cls = _r.choice(classes)
            label = classes.index(cls)
            ranges = self.NUTRIENT_RANGES[cls]
            nutrients = [_r.uniform(r[0], r[1]) for r in ranges]
            self.samples.append((cls, label, nutrients))
        self.img_size = img_size

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        cls, label, nutrients = self.samples[idx]
        img = torch.randn(3, self.img_size, self.img_size)  # synthetic image
        return img, label, torch.tensor(nutrients, dtype=torch.float32)
