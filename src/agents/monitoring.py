"""
AgHealth+ — Monitoring Agent
================================
PRA agent for continuous health monitoring.

Perception : wearables (HR, SpO2, steps, glucose), kitchen sensors, intake log
Reasoning  : GRU-based anomaly detection + skipped-meal detection
Action     : adaptive advice / escalation alert / optional caregiver notification
Feedback   : caregiver acknowledgment → sensitivity adjustment
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from loguru import logger

from .base_agent import BaseAgent
from ..utils.helpers import now_utc


# ──────────────────────────────────────────────────────────────────────────────
# Vital sign normal ranges
# ──────────────────────────────────────────────────────────────────────────────

VITAL_RANGES = {
    "heart_rate":  (50, 100),
    "spo2":        (94, 100),
    "steps_daily": (0, 20000),
    "glucose":     (70, 180),
}


# ──────────────────────────────────────────────────────────────────────────────
# Simulated GRU Anomaly Detector
# (Production: GRU implemented in models/gru_anomaly.py with PyTorch)
# ──────────────────────────────────────────────────────────────────────────────

class SimulatedGRUDetector:
    """
    Simulates a trained GRU model that scores a 24-step vital sequence
    for anomaly probability.

    The simulation uses a rule-based heuristic that matches the paper's
    reported threshold (0.65) and alert sensitivity.

    In production, replace with the trained GRUAnomalyModel from
    src/models/gru_anomaly.py.
    """

    def __init__(self, threshold: float = 0.65):
        self.threshold = threshold
        # Simulated per-user baseline (personalised normal range)
        self._baselines: Dict[str, Dict[str, float]] = {}

    def set_baseline(self, user_id: str, sequence: List[Dict[str, float]]) -> None:
        """Learn per-user baseline from normal-period data."""
        if not sequence:
            return
        self._baselines[user_id] = {
            key: float(np.mean([s[key] for s in sequence if key in s]))
            for key in ["heart_rate", "spo2", "glucose", "steps"]
        }

    def score(
        self,
        user_id: str,
        current_vitals: Dict[str, float],
        sequence: List[Dict[str, float]],
    ) -> Tuple[float, List[str]]:
        """
        Returns (anomaly_score in [0,1], list_of_triggered_signals).
        """
        triggered: List[str] = []
        scores: List[float] = []

        baseline = self._baselines.get(user_id, {})

        for vital, (lo, hi) in VITAL_RANGES.items():
            val = current_vitals.get(vital, None)
            if val is None:
                continue

            # Range violation
            if val < lo or val > hi:
                dev = abs(val - np.clip(val, lo, hi)) / max((hi - lo), 1)
                scores.append(min(1.0, dev * 2))
                triggered.append(f"{vital}={val:.1f} (range [{lo},{hi}])")
            elif vital in baseline:
                # Deviation from personal baseline
                b = baseline[vital]
                dev = abs(val - b) / max(abs(b), 1)
                if dev > 0.20:
                    scores.append(min(0.9, dev))
                    triggered.append(f"{vital} deviated {dev*100:.0f}% from baseline")
                else:
                    scores.append(0.0)
            else:
                scores.append(0.0)

        anomaly_score = float(np.mean(scores)) if scores else 0.0
        return round(anomaly_score, 3), triggered

    def is_anomalous(self, score: float) -> bool:
        return score >= self.threshold


# ──────────────────────────────────────────────────────────────────────────────
# Monitoring Agent
# ──────────────────────────────────────────────────────────────────────────────

class MonitoringAgent(BaseAgent):
    """
    Processes wearable + kitchen sensor streams to detect health anomalies
    and trigger adaptive interventions.
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__("monitoring", config)
        vital_cfg = config.get("vitals", {})
        self.gru = SimulatedGRUDetector(
            threshold=config.get("anomaly_threshold", 0.65)
        )
        self.caregiver_threshold = config.get("caregiver_escalation_risk_level", 0.8)
        self.alert_cooldown = config.get("alert_cooldown_minutes", 60)
        self._alert_times: Dict[str, Any] = {}

    # ── PRA: Perceive ─────────────────────────────────────────────────────────

    def perceive(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "user_id": inputs.get("user_profile", {}).get("user_id", "anon"),
            "user_profile": inputs.get("user_profile", {}),
            "current_vitals": inputs.get("vitals", {}),
            "vital_sequence": inputs.get("vital_sequence", []),  # 24h history
            "intake_log": inputs.get("intake_log", []),
            "context": inputs.get("context", {}),
            "wearables_connected": inputs.get("wearables_connected", True),
        }

    # ── PRA: Reason ──────────────────────────────────────────────────────────

    def reason(self, percept: Dict[str, Any]) -> Dict[str, Any]:
        uid = percept["user_id"]
        vitals = percept["current_vitals"]
        sequence = percept["vital_sequence"]

        # Fallback when wearables are disconnected
        if not percept["wearables_connected"]:
            vitals = self._infer_from_intake(percept["intake_log"])
            logger.warning("MonitoringAgent | wearable offline — using intake proxy | user={}", uid[:8])

        # Update personalised baseline (first 7 days = warm-up)
        if len(sequence) >= 7:
            self.gru.set_baseline(uid, sequence[-168:])  # last 7 days × 24h

        # Anomaly detection
        risk_score, triggers = self.gru.score(uid, vitals, sequence)
        is_anomaly = self.gru.is_anomalous(risk_score)

        # Skipped meal detection
        skipped_meals = self._detect_skipped_meals(percept["intake_log"], percept["context"])

        return {
            "risk_score": risk_score,
            "is_anomaly": is_anomaly,
            "triggers": triggers,
            "skipped_meals": skipped_meals,
            "vitals": vitals,
            "user_profile": percept["user_profile"],
            "uid": uid,
        }

    # ── PRA: Act ─────────────────────────────────────────────────────────────

    def act(self, decision: Dict[str, Any]) -> Dict[str, Any]:
        uid = decision["uid"]
        risk_score = decision["risk_score"]
        triggers = decision["triggers"]
        profile = decision["user_profile"]
        skipped = decision["skipped_meals"]

        # Cooldown check
        from datetime import datetime, timedelta
        last_alert = self._alert_times.get(uid)
        if last_alert and (datetime.now() - last_alert).total_seconds() / 60 < self.alert_cooldown:
            return {
                "action_type": "monitoring_suppressed",
                "message": "Monitoring alert suppressed (cooldown active).",
                "risk_score": risk_score,
                "_confidence": 1.0,
                "_explanation_hint": "Alert cooldown active — preventing over-notification.",
            }

        actions: List[str] = []
        caregiver_notify = False

        if decision["is_anomaly"]:
            actions.append(self._build_alert(triggers, risk_score, profile))
            if risk_score >= self.caregiver_threshold:
                caregiver_notify = True
                actions.append("[CAREGIVER NOTIFIED] High-risk anomaly detected.")
            self._alert_times[uid] = datetime.now()

        if skipped:
            actions.append(f"Skipped meal(s) detected: {', '.join(skipped)}. Consider a light snack.")

        if not actions:
            actions.append("All vitals within normal range. Keep it up! 🌿")

        return {
            "action_type": "alert" if decision["is_anomaly"] else "monitoring_ok",
            "message": " ".join(actions),
            "risk_score": risk_score,
            "triggers": triggers,
            "caregiver_notified": caregiver_notify,
            "skipped_meals": skipped,
            "_confidence": 1.0,
            "_explanation_hint": (
                f"GRU anomaly score={risk_score:.3f} (threshold={self.gru.threshold}). "
                f"Triggers: {triggers}."
            ),
        }

    # ── Helpers ──────────────────────────────────────────────────────────────

    @staticmethod
    def _build_alert(triggers: List[str], risk_score: float, profile: Dict) -> str:
        if risk_score > 0.85:
            severity = "⚠️ URGENT"
        elif risk_score > 0.65:
            severity = "⚠️ Attention"
        else:
            severity = "ℹ️ Notice"

        name = profile.get("name", "")
        trigger_str = "; ".join(triggers[:3]) if triggers else "vital deviation detected"
        return f"{severity}: {name + ' — ' if name else ''}{trigger_str}."

    @staticmethod
    def _detect_skipped_meals(intake_log: List[Dict], context: Dict) -> List[str]:
        """Identify meal slots with no intake logged."""
        hour = context.get("hour", 12)
        expected_slots = []
        if hour >= 10:
            expected_slots.append("breakfast")
        if hour >= 14:
            expected_slots.append("lunch")
        if hour >= 20:
            expected_slots.append("dinner")

        logged_slots = {entry.get("meal_type", "").lower() for entry in intake_log}
        return [s for s in expected_slots if s not in logged_slots]

    @staticmethod
    def _infer_from_intake(intake_log: List[Dict]) -> Dict[str, float]:
        """Proxy vitals estimation when wearables are offline."""
        n_meals = len(intake_log)
        calories_today = sum(e.get("calories", 0) for e in intake_log)
        return {
            "heart_rate": 72.0,     # default resting
            "spo2": 98.0,
            "steps": max(0, 2000 - n_meals * 100),
            "glucose": 100.0 + max(0, calories_today - 1500) * 0.02,
        }
