"""
AgHealth+ — Reminder Agent
============================
PRA agent that schedules personalised, low-friction reminders using a
contextual bandit (UCB1 variant) as described in the paper's Algorithm 2.

Perception : engagement logs, sleep windows, device presence, missed events
Reasoning  : contextual bandit selects (time, modality) with highest expected reward
Action     : vibration / text banner / icon / sound nudge
Feedback   : comply → +1, snooze → 0, ignore → –1 → bandit arm update
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from loguru import logger

from .base_agent import BaseAgent


# ──────────────────────────────────────────────────────────────────────────────
# Contextual Bandit — UCB1
# ──────────────────────────────────────────────────────────────────────────────

MODALITIES = ["vibration", "text_banner", "icon", "sound"]
TIME_SLOTS = ["morning", "midday", "afternoon", "evening"]

N_ARMS = len(MODALITIES) * len(TIME_SLOTS)  # 16 arms


@dataclass
class BanditArm:
    modality: str
    time_slot: str
    n_pulls: int = 0
    total_reward: float = 0.0

    @property
    def mean_reward(self) -> float:
        return self.total_reward / self.n_pulls if self.n_pulls > 0 else 0.0

    def ucb_score(self, total_pulls: int, c: float = 1.0) -> float:
        if self.n_pulls == 0:
            return float("inf")
        exploration = c * math.sqrt(math.log(total_pulls) / self.n_pulls)
        return self.mean_reward + exploration

    def update(self, reward: float) -> None:
        self.n_pulls += 1
        self.total_reward += reward


class ContextualBandit:
    """UCB1 bandit over (modality × time_slot) arms."""

    def __init__(self, exploration_param: float = 1.0):
        self.c = exploration_param
        self.arms: List[BanditArm] = [
            BanditArm(modality=m, time_slot=t)
            for t in TIME_SLOTS for m in MODALITIES
        ]
        self.total_pulls = 0

    def select(self, context: Dict[str, Any]) -> BanditArm:
        """Select best arm given current context (time slot filtering)."""
        current_slot = self._time_to_slot(context.get("hour", 12))
        # Filter arms relevant to current time (prefer matching slots)
        preferred = [a for a in self.arms if a.time_slot == current_slot]
        candidates = preferred if preferred else self.arms
        return max(candidates, key=lambda a: a.ucb_score(max(self.total_pulls, 1), self.c))

    def update(self, arm: BanditArm, reward: float) -> None:
        arm.update(reward)
        self.total_pulls += 1

    @staticmethod
    def _time_to_slot(hour: int) -> str:
        if hour < 10:
            return "morning"
        elif hour < 13:
            return "midday"
        elif hour < 17:
            return "afternoon"
        return "evening"

    def best_arms(self, n: int = 3) -> List[BanditArm]:
        return sorted(self.arms, key=lambda a: -a.mean_reward)[:n]


# ──────────────────────────────────────────────────────────────────────────────
# Reminder Agent
# ──────────────────────────────────────────────────────────────────────────────

class ReminderAgent(BaseAgent):
    """
    Adaptive reminder scheduling agent.

    Uses UCB1 contextual bandit to optimise (time_slot, modality) pairs
    for each user, minimising alert fatigue while maximising adherence.
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__("reminder", config)
        self.bandit = ContextualBandit(
            exploration_param=config.get("exploration_param", 1.0)
        )
        self.max_daily = config.get("max_daily_reminders", 5)
        self.min_gap_min = config.get("min_gap_minutes", 30)
        self.low_stimulation = config.get("low_stimulation", True)
        self._daily_count: Dict[str, int] = {}     # {user_id: count}
        self._last_sent: Dict[str, datetime] = {}  # {user_id: last_time}

    # ── PRA: Perceive ─────────────────────────────────────────────────────────

    def perceive(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "user_id": inputs.get("user_profile", {}).get("user_id", "anon"),
            "context": inputs.get("context", {}),
            "engagement_history": inputs.get("engagement_history", []),
            "missed_events": inputs.get("missed_events", []),
            "device_present": inputs.get("device_present", True),
            "sleep_window": inputs.get("sleep_window", {"start": 23, "end": 7}),
            "user_profile": inputs.get("user_profile", {}),
        }

    # ── PRA: Reason ──────────────────────────────────────────────────────────

    def reason(self, percept: Dict[str, Any]) -> Dict[str, Any]:
        uid = percept["user_id"]
        context = percept["context"]
        hour = context.get("hour", 12)

        # Suppress during sleep
        sleep = percept["sleep_window"]
        in_sleep = (hour >= sleep["start"] or hour < sleep["end"])
        if in_sleep:
            return {"send": False, "reason": "User in sleep window", "percept": percept}

        # Respect max daily budget
        daily_count = self._daily_count.get(uid, 0)
        if daily_count >= self.max_daily:
            return {"send": False, "reason": "Daily reminder budget exhausted", "percept": percept}

        # Respect minimum gap
        last = self._last_sent.get(uid)
        if last:
            gap = (datetime.now() - last).total_seconds() / 60
            if gap < self.min_gap_min:
                return {"send": False, "reason": f"Too soon (gap={gap:.0f}m)", "percept": percept}

        # Select best arm
        arm = self.bandit.select(context)

        # Low-stimulation override for neurodivergent users
        if self.low_stimulation:
            disability = percept["user_profile"].get("neurodivergent_type", "none")
            if disability in ["ASD", "ADHD"] and arm.modality == "sound":
                arm = next((a for a in self.bandit.arms if a.modality == "vibration"), arm)

        return {"send": True, "arm": arm, "percept": percept, "reason": "Bandit selected arm"}

    # ── PRA: Act ─────────────────────────────────────────────────────────────

    def act(self, decision: Dict[str, Any]) -> Dict[str, Any]:
        if not decision.get("send"):
            return {
                "action_type": "reminder_suppressed",
                "message": decision.get("reason", "Reminder not sent."),
                "_confidence": 1.0,
                "_explanation_hint": decision.get("reason", ""),
            }

        arm: BanditArm = decision["arm"]
        percept = decision["percept"]
        uid = percept["user_id"]
        profile = percept["user_profile"]

        # Build message
        message = self._build_message(arm.modality, profile, percept["missed_events"])

        # Update state
        self._daily_count[uid] = self._daily_count.get(uid, 0) + 1
        self._last_sent[uid] = datetime.now()

        return {
            "action_type": "reminder",
            "message": message,
            "modality": arm.modality,
            "time_slot": arm.time_slot,
            "scheduled_at": self._next_slot_time(arm.time_slot),
            "_confidence": min(1.0, arm.mean_reward + 0.5),
            "_explanation_hint": f"Bandit arm ({arm.modality}, {arm.time_slot}) selected; "
                                  f"mean_reward={arm.mean_reward:.2f}, n_pulls={arm.n_pulls}",
            "_arm_ref": arm,  # for feedback update
        }

    # ── Feedback ─────────────────────────────────────────────────────────────

    def ingest_feedback(self, feedback: Dict[str, Any]) -> None:
        super().ingest_feedback(feedback)
        arm: Optional[BanditArm] = feedback.get("_arm_ref")
        response = feedback.get("response", "ignored")  # "complied" | "snoozed" | "ignored"
        reward_map = {"complied": 1.0, "snoozed": 0.0, "ignored": -1.0}
        reward = reward_map.get(response, 0.0)
        if arm:
            self.bandit.update(arm, reward)
            logger.debug("ReminderAgent | bandit update | arm=({},{}) reward={}", arm.modality, arm.time_slot, reward)

    # ── Helpers ──────────────────────────────────────────────────────────────

    @staticmethod
    def _build_message(modality: str, profile: Dict, missed: List) -> str:
        name = profile.get("name", "there")
        base = f"Hi {name}! "
        if missed:
            base += f"You missed: {missed[0]}. "
        base += "Time for your meal / medication check."
        if modality == "vibration":
            return f"[VIBRATION] {base}"
        elif modality == "text_banner":
            return f"[BANNER 📋] {base}"
        elif modality == "icon":
            return f"[ICON 🍽️] {base}"
        else:
            return f"[SOUND 🔔] {base}"

    @staticmethod
    def _next_slot_time(time_slot: str) -> str:
        slot_hours = {"morning": 8, "midday": 12, "afternoon": 15, "evening": 18}
        hour = slot_hours.get(time_slot, 12)
        now = datetime.now()
        scheduled = now.replace(hour=hour, minute=0, second=0, microsecond=0)
        if scheduled < now:
            scheduled += timedelta(days=1)
        return scheduled.isoformat()
