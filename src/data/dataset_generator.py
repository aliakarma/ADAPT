"""
AgHealth+ — Synthetic Dataset Generator
==========================================
Generates a clinically plausible dataset of 500 simulated users
across 8 weeks as described in the paper's Section 5.4.

Output
------
data/synthetic/
    users.json           — user profiles (demographics, conditions, preferences)
    daily_records.json   — per-user daily intake + vitals + reminder responses
    weekly_summary.json  — weekly aggregation per user

Probabilistic modeling ensures diversity while maintaining
clinical realism and strict data confidentiality (no real PII).
"""
from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from ..utils.helpers import set_global_seed, save_json, now_utc


# ──────────────────────────────────────────────────────────────────────────────
# User Profile
# ──────────────────────────────────────────────────────────────────────────────

CLINICAL_PROFILES = ["diabetes", "hypertension", "mixed_cardiometabolic"]
NEURODIVERGENT_TYPES = ["ASD", "ADHD", "none"]
DISABILITY_TYPES = ["physical", "sensory", "cognitive", "none"]
CULTURAL_PREFS = ["standard", "halal", "vegetarian", "vegan", "low_spice"]
SENSORY_PREFS = ["mild", "soft", "spiced", "mixed"]
READING_LEVELS = ["simple", "standard", "clinical"]
GENDERS = ["male", "female", "non_binary"]


@dataclass
class UserProfile:
    user_id: str
    age: int
    gender: str
    clinical_profile: str
    conditions: List[str]
    neurodivergent_type: str
    disability_type: str
    cultural_preference: str
    sensory_preference: str
    reading_level: str
    daily_calorie_target: int
    low_sodium: bool
    calorie_restriction: bool
    allergies: List[str]
    caregiver_enabled: bool
    medication_schedule: List[str]  # meal-relative timing
    name: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class DailyRecord:
    user_id: str
    day: int
    week: int
    date_str: str
    meals_consumed: List[Dict[str, Any]]
    calories_total: float
    protein_g: float
    fat_g: float
    sugar_g: float
    sodium_mg: float
    vitals: Dict[str, float]
    reminder_responses: List[str]   # "complied" | "snoozed" | "ignored"
    adherence_score: float          # 0–1
    anomaly_flagged: bool
    mood: str                       # "good" | "neutral" | "stressed"
    notes: str = ""


# ──────────────────────────────────────────────────────────────────────────────
# Generator
# ──────────────────────────────────────────────────────────────────────────────

class SyntheticDatasetGenerator:
    """
    Generates the full pilot simulation dataset.

    Parameters
    ----------
    n_users : int
        Number of simulated users (paper: 500).
    n_weeks : int
        Simulation duration in weeks (paper: 8).
    seed : int
        Random seed for full reproducibility.
    """

    ALLERGY_POOL = ["nuts", "gluten", "dairy", "shellfish", "eggs", "soy"]
    MEAL_NAMES = [
        "Lentil soup", "Grilled chicken salad", "Quinoa bowl", "Rice & beans",
        "Oatmeal + berries", "Egg omelette", "Baked salmon", "Veggie stir-fry",
        "Turkey wrap", "Chickpea curry", "Greek yogurt + nuts", "Pumpkin soup",
    ]

    def __init__(self, n_users: int = 500, n_weeks: int = 8, seed: int = 42):
        self.n_users = n_users
        self.n_weeks = n_weeks
        self.seed = seed
        set_global_seed(seed)

    def generate(self, output_dir: str = "data/synthetic") -> Dict[str, Any]:
        """Run full generation and save outputs."""
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        print(f"Generating {self.n_users} user profiles...")
        users = [self._generate_user(i) for i in range(self.n_users)]

        print(f"Generating {self.n_users * self.n_weeks * 7} daily records...")
        daily_records: List[DailyRecord] = []
        for user in users:
            for week in range(1, self.n_weeks + 1):
                for day_in_week in range(1, 8):
                    day = (week - 1) * 7 + day_in_week
                    record = self._generate_daily_record(user, day, week)
                    daily_records.append(record)

        # Weekly summaries
        weekly = self._aggregate_weekly(users, daily_records)

        # Save
        save_json([u.to_dict() for u in users], f"{output_dir}/users.json")
        save_json([self._record_to_dict(r) for r in daily_records], f"{output_dir}/daily_records.json")
        save_json(weekly, f"{output_dir}/weekly_summary.json")

        stats = self._dataset_stats(users, daily_records)
        save_json(stats, f"{output_dir}/dataset_stats.json")

        print(f"Dataset saved to {output_dir}/")
        return stats

    # ── User generation ───────────────────────────────────────────────────────

    def _generate_user(self, idx: int) -> UserProfile:
        rng = np.random

        profile = rng.choice(CLINICAL_PROFILES, p=[0.4, 0.35, 0.25])
        conditions = self._conditions_from_profile(profile)

        age = int(rng.normal(45, 15))
        age = max(18, min(85, age))

        neuro = rng.choice(NEURODIVERGENT_TYPES, p=[0.15, 0.15, 0.70])
        disability = rng.choice(DISABILITY_TYPES, p=[0.20, 0.15, 0.15, 0.50])

        calorie_target = int(rng.normal(2000, 300))
        calorie_target = max(1400, min(3000, calorie_target))

        n_allergies = rng.choice([0, 1, 2], p=[0.70, 0.20, 0.10])
        allergies = list(rng.choice(self.ALLERGY_POOL, size=n_allergies, replace=False))

        return UserProfile(
            user_id=f"user_{idx:04d}",
            age=age,
            gender=rng.choice(GENDERS, p=[0.48, 0.48, 0.04]),
            clinical_profile=profile,
            conditions=conditions,
            neurodivergent_type=neuro,
            disability_type=disability,
            cultural_preference=rng.choice(CULTURAL_PREFS, p=[0.5, 0.2, 0.15, 0.05, 0.10]),
            sensory_preference=rng.choice(SENSORY_PREFS, p=[0.5, 0.2, 0.2, 0.1]),
            reading_level=rng.choice(READING_LEVELS,
                                      p=[0.30, 0.45, 0.25]),
            daily_calorie_target=calorie_target,
            low_sodium="hypertension" in conditions or profile == "mixed_cardiometabolic",
            calorie_restriction=bool(rng.choice([True, False], p=[0.35, 0.65])),
            allergies=allergies,
            caregiver_enabled=bool(rng.choice([True, False], p=[0.40, 0.60])),
            medication_schedule=list(
                rng.choice(
                    ["with_breakfast", "with_dinner", "both"],
                    p=[0.4, 0.3, 0.3]
                ).split("_and_") if False else
                [["with_breakfast"], ["with_dinner"], ["with_breakfast", "with_dinner"]][
                    int(rng.choice(3, p=[0.4, 0.3, 0.3]))
                ]
            ),
            name=f"User{idx:04d}",
        )

    @staticmethod
    def _conditions_from_profile(profile: str) -> List[str]:
        if profile == "diabetes":
            return ["diabetes"]
        elif profile == "hypertension":
            return ["hypertension"]
        else:
            return ["diabetes", "hypertension"]

    # ── Daily record generation ───────────────────────────────────────────────

    def _generate_daily_record(
        self, user: UserProfile, day: int, week: int
    ) -> DailyRecord:
        rng = np.random

        # Base adherence improves over time (system effect simulation)
        # Paper: adherence 54% → 81% with adaptive reminders
        base_adherence = 0.54 + 0.27 * (day / (self.n_weeks * 7))
        adherence = float(np.clip(rng.normal(base_adherence, 0.12), 0, 1))

        n_meals = 3
        meals = []
        total_cal = total_prot = total_fat = total_sug = total_sod = 0.0

        for _ in range(n_meals):
            meal_name = rng.choice(self.MEAL_NAMES)
            cal = float(rng.normal(500, 80))
            prot = float(rng.normal(20, 6))
            fat = float(rng.normal(15, 5))
            sug = float(rng.normal(8, 4))
            sod = float(rng.normal(400, 100))
            if user.low_sodium:
                sod *= 0.7
            meals.append({"name": meal_name, "calories": round(cal, 1),
                           "protein_g": round(prot, 1), "fat_g": round(fat, 1)})
            total_cal += cal; total_prot += prot; total_fat += fat
            total_sug += sug; total_sod += sod

        # Generate vitals
        vitals = self._gen_vitals(user, adherence)
        anomaly = vitals["heart_rate"] > 110 or vitals["glucose"] > 200 or vitals["spo2"] < 92

        # Reminder responses — improve with adaptive system
        n_reminders = rng.randint(2, 6)
        responses = []
        for _ in range(n_reminders):
            comply_prob = adherence * 0.9
            r = rng.random()
            responses.append("complied" if r < comply_prob else ("snoozed" if r < comply_prob + 0.1 else "ignored"))

        mood = rng.choice(["good", "neutral", "stressed"], p=[0.45, 0.35, 0.20])

        return DailyRecord(
            user_id=user.user_id,
            day=day,
            week=week,
            date_str=f"W{week:02d}D{day % 7 + 1}",
            meals_consumed=meals,
            calories_total=round(total_cal, 1),
            protein_g=round(total_prot, 1),
            fat_g=round(total_fat, 1),
            sugar_g=round(total_sug, 1),
            sodium_mg=round(total_sod, 1),
            vitals=vitals,
            reminder_responses=responses,
            adherence_score=round(adherence, 3),
            anomaly_flagged=anomaly,
            mood=mood,
        )

    @staticmethod
    def _gen_vitals(user: UserProfile, adherence: float) -> Dict[str, float]:
        rng = np.random
        hr_base = 72
        glucose_base = 90 if "diabetes" not in user.conditions else 110
        spo2_base = 98

        # Stress / mood injection
        stress = rng.normal(0, 5)

        hr = float(np.clip(rng.normal(hr_base + stress, 8), 45, 140))
        spo2 = float(np.clip(rng.normal(spo2_base, 1.2), 88, 100))
        glucose = float(np.clip(rng.normal(glucose_base + (1 - adherence) * 40, 15), 60, 280))
        steps = float(np.clip(rng.normal(5000, 1500), 0, 20000))
        hydration = float(np.clip(rng.normal(0.75, 0.15), 0, 1))

        return {
            "heart_rate": round(hr, 1),
            "spo2": round(spo2, 1),
            "glucose": round(glucose, 1),
            "steps": round(steps),
            "hydration_score": round(hydration, 3),
        }

    # ── Aggregation ───────────────────────────────────────────────────────────

    def _aggregate_weekly(
        self, users: List[UserProfile], records: List[DailyRecord]
    ) -> List[Dict[str, Any]]:
        from collections import defaultdict
        weekly: Dict[str, Dict[int, List[DailyRecord]]] = defaultdict(lambda: defaultdict(list))
        for r in records:
            weekly[r.user_id][r.week].append(r)

        summaries = []
        for user in users:
            for week, recs in weekly[user.user_id].items():
                n = len(recs)
                avg_adherence = np.mean([r.adherence_score for r in recs])
                avg_cal = np.mean([r.calories_total for r in recs])
                n_anomalies = sum(1 for r in recs if r.anomaly_flagged)
                all_responses = [resp for r in recs for resp in r.reminder_responses]
                comply_rate = all_responses.count("complied") / len(all_responses) if all_responses else 0
                summaries.append({
                    "user_id": user.user_id,
                    "week": week,
                    "avg_adherence": round(float(avg_adherence), 3),
                    "avg_calories": round(float(avg_cal), 1),
                    "n_anomalies": n_anomalies,
                    "reminder_compliance_rate": round(comply_rate, 3),
                    "n_days": n,
                })
        return summaries

    # ── Stats ─────────────────────────────────────────────────────────────────

    def _dataset_stats(self, users, records) -> Dict[str, Any]:
        adherences = [r.adherence_score for r in records]
        anomalies = sum(1 for r in records if r.anomaly_flagged)
        return {
            "n_users": len(users),
            "n_weeks": self.n_weeks,
            "n_records": len(records),
            "adherence_mean": round(float(np.mean(adherences)), 3),
            "adherence_std": round(float(np.std(adherences)), 3),
            "anomaly_rate": round(anomalies / len(records), 3),
            "clinical_profiles": {
                p: sum(1 for u in users if u.clinical_profile == p)
                for p in CLINICAL_PROFILES
            },
            "neurodivergent": {
                t: sum(1 for u in users if u.neurodivergent_type == t)
                for t in NEURODIVERGENT_TYPES
            },
            "disability": {
                t: sum(1 for u in users if u.disability_type == t)
                for t in DISABILITY_TYPES
            },
        }

    @staticmethod
    def _record_to_dict(r: DailyRecord) -> Dict[str, Any]:
        return {
            "user_id": r.user_id,
            "day": r.day, "week": r.week, "date_str": r.date_str,
            "meals_consumed": r.meals_consumed,
            "calories_total": r.calories_total,
            "protein_g": r.protein_g, "fat_g": r.fat_g,
            "sugar_g": r.sugar_g, "sodium_mg": r.sodium_mg,
            "vitals": r.vitals,
            "reminder_responses": r.reminder_responses,
            "adherence_score": r.adherence_score,
            "anomaly_flagged": r.anomaly_flagged,
            "mood": r.mood,
        }
