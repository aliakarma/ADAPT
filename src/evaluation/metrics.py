"""
AgHealth+ — Evaluation Metrics
==================================
Computes all evaluation metrics as defined in Section 5.7 of the paper:

1. Nutritional Adequacy      — % of daily plans meeting DRI guidelines
2. Adherence Rate            — ratio of complied reminders to total
3. User Satisfaction Score   — simulated Likert-scale mean (paper: 4.2/5)
4. Explainability Rate       — % of decisions with plain-language explanation
5. Caregiver Burden          — reduction in caregiver interventions

Plus model-level metrics:
- Food Recognition Accuracy  (paper: 99%)
- Nutrient Prediction MAE    (paper: cal=13.9, prot=1.2, fat=1.3, sugar=1.1)
"""
from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..utils.helpers import DRI, nutrition_score, save_json


# ──────────────────────────────────────────────────────────────────────────────
# Nutritional Adequacy
# ──────────────────────────────────────────────────────────────────────────────

def compute_nutritional_adequacy(
    daily_records: List[Dict[str, Any]],
    dri: Optional[Dict[str, float]] = None,
    threshold: float = 0.80,
) -> Dict[str, float]:
    """
    Percentage of daily records that meet >= `threshold` DRI coverage.
    Paper baseline comparison: system 27% better than manual planning.
    """
    dri = dri or DRI
    meeting = 0
    scores = []

    for record in daily_records:
        nutrients = {
            "calories":   record.get("calories_total", 0),
            "protein_g":  record.get("protein_g", 0),
            "fat_g":      record.get("fat_g", 0),
            "sugar_g":    record.get("sugar_g", 0),
        }
        score = nutrition_score(nutrients, dri)
        scores.append(score)
        if score >= threshold:
            meeting += 1

    adequacy_pct = meeting / len(daily_records) if daily_records else 0.0
    return {
        "nutritional_adequacy_pct": round(adequacy_pct * 100, 2),
        "mean_nutrition_score": round(float(np.mean(scores)), 4),
        "std_nutrition_score": round(float(np.std(scores)), 4),
        "n_records": len(daily_records),
        "threshold_used": threshold,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Adherence Rate
# ──────────────────────────────────────────────────────────────────────────────

def compute_adherence_rate(
    daily_records: List[Dict[str, Any]],
) -> Dict[str, float]:
    """
    Ratio of 'complied' reminder responses to total responses.
    Paper: 54% (static) → 81% (adaptive).
    """
    all_responses: List[str] = []
    for record in daily_records:
        all_responses.extend(record.get("reminder_responses", []))

    if not all_responses:
        return {"adherence_rate_pct": 0.0, "n_reminders": 0}

    complied = all_responses.count("complied")
    snoozed = all_responses.count("snoozed")
    ignored = all_responses.count("ignored")
    rate = complied / len(all_responses)

    return {
        "adherence_rate_pct": round(rate * 100, 2),
        "n_complied": complied,
        "n_snoozed": snoozed,
        "n_ignored": ignored,
        "n_reminders": len(all_responses),
    }


# ──────────────────────────────────────────────────────────────────────────────
# User Satisfaction (Simulated Likert)
# ──────────────────────────────────────────────────────────────────────────────

def compute_user_satisfaction(
    daily_records: List[Dict[str, Any]],
    seed: int = 42,
) -> Dict[str, float]:
    """
    Simulate a Likert-scale (1–5) satisfaction survey.
    Higher adherence and no-anomaly days correlate with higher satisfaction.
    Paper result: mean = 4.2 / 5.
    """
    rng = np.random.default_rng(seed)
    scores = []
    for record in daily_records:
        adherence = record.get("adherence_score", 0.7)
        anomaly = int(record.get("anomaly_flagged", False))
        mood_bonus = {"good": 0.3, "neutral": 0.0, "stressed": -0.3}.get(record.get("mood", "neutral"), 0.0)

        base = 3.5 + adherence * 1.2 - anomaly * 0.6 + mood_bonus
        score = float(np.clip(rng.normal(base, 0.3), 1, 5))
        scores.append(score)

    mean_score = float(np.mean(scores))
    return {
        "satisfaction_mean": round(mean_score, 3),
        "satisfaction_std": round(float(np.std(scores)), 3),
        "satisfaction_min": round(float(np.min(scores)), 2),
        "satisfaction_max": round(float(np.max(scores)), 2),
        "n_responses": len(scores),
    }


# ──────────────────────────────────────────────────────────────────────────────
# Explainability Rate
# ──────────────────────────────────────────────────────────────────────────────

def compute_explainability_rate(
    responses: List[Dict[str, Any]],
) -> Dict[str, float]:
    """
    Percentage of system decisions that generated a non-empty explanation.
    Paper: 92%.
    """
    if not responses:
        return {"explainability_rate_pct": 0.0, "n_decisions": 0}

    explained = sum(
        1 for r in responses
        if r.get("explanation") and len(str(r["explanation"])) > 10
    )
    rate = explained / len(responses)
    return {
        "explainability_rate_pct": round(rate * 100, 2),
        "n_explained": explained,
        "n_decisions": len(responses),
    }


# ──────────────────────────────────────────────────────────────────────────────
# Caregiver Burden Reduction
# ──────────────────────────────────────────────────────────────────────────────

def compute_caregiver_burden_reduction(
    daily_records: List[Dict[str, Any]],
    baseline_intervention_rate: float = 0.15,
) -> Dict[str, float]:
    """
    Estimate reduction in caregiver interventions.
    Paper: 35% reduction.
    High-adherence, low-anomaly days require fewer caregiver interventions.
    """
    total_days = len(daily_records)
    actual_interventions = sum(
        1 for r in daily_records
        if r.get("anomaly_flagged") or r.get("adherence_score", 1) < 0.4
    )
    baseline = int(total_days * baseline_intervention_rate)
    actual_rate = actual_interventions / total_days if total_days else 0
    reduction = max(0.0, (baseline_intervention_rate - actual_rate) / baseline_intervention_rate)

    return {
        "baseline_intervention_rate": baseline_intervention_rate,
        "actual_intervention_rate": round(actual_rate, 4),
        "caregiver_burden_reduction_pct": round(reduction * 100, 2),
        "n_interventions": actual_interventions,
        "n_days": total_days,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Food Recognition Accuracy (Model-level)
# ──────────────────────────────────────────────────────────────────────────────

def compute_food_recognition_accuracy(
    predictions: List[str],
    ground_truth: List[str],
) -> Dict[str, float]:
    """
    Top-1 accuracy for food classification.
    Paper: 99% on 10-class evaluation dataset.
    """
    assert len(predictions) == len(ground_truth), "Length mismatch"
    correct = sum(p == g for p, g in zip(predictions, ground_truth))
    acc = correct / len(predictions) if predictions else 0.0

    # Per-class accuracy
    classes = list(set(ground_truth))
    per_class: Dict[str, float] = {}
    for cls in classes:
        idxs = [i for i, g in enumerate(ground_truth) if g == cls]
        cls_correct = sum(1 for i in idxs if predictions[i] == cls)
        per_class[cls] = round(cls_correct / len(idxs), 4) if idxs else 0.0

    return {
        "accuracy": round(acc * 100, 2),
        "n_correct": correct,
        "n_total": len(predictions),
        "per_class_accuracy": per_class,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Nutrient Prediction MAE
# ──────────────────────────────────────────────────────────────────────────────

def compute_nutrient_mae(
    predictions: List[Dict[str, float]],
    ground_truth: List[Dict[str, float]],
    nutrient_keys: Optional[List[str]] = None,
) -> Dict[str, float]:
    """
    Mean Absolute Error for each nutrient.
    Paper: calories=13.9, protein=1.2, fat=1.3, sugar=1.1.
    """
    nutrient_keys = nutrient_keys or ["calories", "protein_g", "fat_g", "sugar_g"]
    mae: Dict[str, float] = {}
    for key in nutrient_keys:
        errors = [abs(p.get(key, 0) - g.get(key, 0)) for p, g in zip(predictions, ground_truth)]
        mae[f"mae_{key}"] = round(float(np.mean(errors)), 3)
    return mae


# ──────────────────────────────────────────────────────────────────────────────
# Full Evaluation Pipeline
# ──────────────────────────────────────────────────────────────────────────────

def run_full_evaluation(
    daily_records: List[Dict[str, Any]],
    system_responses: Optional[List[Dict[str, Any]]] = None,
    food_preds: Optional[Tuple[List[str], List[str]]] = None,
    nutrient_preds: Optional[Tuple[List[Dict], List[Dict]]] = None,
    output_path: str = "results/tables/evaluation_results.json",
) -> Dict[str, Any]:
    """
    Run all evaluation metrics and save to results/tables/.
    """
    results: Dict[str, Any] = {}

    results["nutritional_adequacy"] = compute_nutritional_adequacy(daily_records)
    results["adherence"] = compute_adherence_rate(daily_records)
    results["user_satisfaction"] = compute_user_satisfaction(daily_records)
    results["caregiver_burden"] = compute_caregiver_burden_reduction(daily_records)

    if system_responses:
        results["explainability"] = compute_explainability_rate(system_responses)

    if food_preds:
        preds, gt = food_preds
        results["food_recognition"] = compute_food_recognition_accuracy(preds, gt)

    if nutrient_preds:
        preds, gt = nutrient_preds
        results["nutrient_mae"] = compute_nutrient_mae(preds, gt)

    save_json(results, output_path)
    return results
