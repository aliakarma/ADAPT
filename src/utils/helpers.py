"""
AgHealth+ — General Utility Helpers
"""
import hashlib
import json
import random
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import yaml


# ──────────────────────────────────────────────────────────────────────────────
# Reproducibility
# ──────────────────────────────────────────────────────────────────────────────

def set_global_seed(seed: int = 42) -> None:
    """Set random seeds for full reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


# ──────────────────────────────────────────────────────────────────────────────
# Config loading
# ──────────────────────────────────────────────────────────────────────────────

def load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def load_config(name: str = "system_config") -> Dict[str, Any]:
    base = Path(__file__).resolve().parents[2] / "configs"
    return load_yaml(str(base / f"{name}.yaml"))


# ──────────────────────────────────────────────────────────────────────────────
# Tracing & Identity
# ──────────────────────────────────────────────────────────────────────────────

def generate_trace_id() -> str:
    return str(uuid.uuid4())


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def hash_user_id(user_id: str) -> str:
    """One-way hash for privacy-safe logging."""
    return hashlib.sha256(user_id.encode()).hexdigest()[:12]


# ──────────────────────────────────────────────────────────────────────────────
# Serialisation
# ──────────────────────────────────────────────────────────────────────────────

def to_json(obj: Any, indent: int = 2) -> str:
    def _default(o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, datetime):
            return o.isoformat()
        return str(o)
    return json.dumps(obj, default=_default, indent=indent)


def save_json(obj: Any, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write(to_json(obj))


def load_json(path: str) -> Any:
    with open(path, "r") as f:
        return json.load(f)


# ──────────────────────────────────────────────────────────────────────────────
# Timing
# ──────────────────────────────────────────────────────────────────────────────

class Timer:
    def __init__(self):
        self._start: Optional[float] = None

    def __enter__(self):
        self._start = time.perf_counter()
        return self

    def __exit__(self, *_):
        self.elapsed = time.perf_counter() - self._start

    @property
    def elapsed_ms(self) -> float:
        return self.elapsed * 1000


# ──────────────────────────────────────────────────────────────────────────────
# Nutrition helpers
# ──────────────────────────────────────────────────────────────────────────────

DRI = {
    "calories": 2000,
    "protein_g": 50,
    "fat_g": 65,
    "carbs_g": 300,
    "sugar_g": 50,
    "sodium_mg": 2300,
    "fiber_g": 28,
}


def nutrition_score(meal_nutrients: Dict[str, float],
                    dri: Optional[Dict[str, float]] = None) -> float:
    """
    Returns a score in [0, 1] representing how closely a meal's
    nutritional profile meets DRI targets (simplified adequacy proxy).
    """
    dri = dri or DRI
    scores = []
    for key, target in dri.items():
        val = meal_nutrients.get(key, 0.0)
        ratio = val / target if target > 0 else 0.0
        # Penalise both under- and over-shooting
        scores.append(max(0.0, 1.0 - abs(ratio - 1.0)))
    return float(np.mean(scores))


def clinical_compliance(meal: Dict[str, float],
                         constraints: Dict[str, Any]) -> bool:
    """
    Returns True iff the meal satisfies ALL hard clinical constraints.
    constraints format: {"sodium_mg": {"max": 1500}, "sugar_g": {"max": 30}, ...}
    """
    for nutrient, rule in constraints.items():
        val = meal.get(nutrient, 0.0)
        if "max" in rule and val > rule["max"]:
            return False
        if "min" in rule and val < rule["min"]:
            return False
    return True
