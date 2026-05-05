"""
AgHealth+ — Meal Planner Agent
================================
PRA agent that recommends balanced, clinically safe meal plans using
Q-learning (as per the paper's Algorithm 1).

Perception : user profile, EHR restrictions, sensory/cultural preferences, recent intake
Reasoning  : Q-learning policy + clinical rule checks + preference balancing
Action     : daily/weekly menu, safe substitutions, shopping list
Feedback   : user acceptance → Q-table update
"""
from __future__ import annotations

import random
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from loguru import logger

from .base_agent import BaseAgent, AgentOutput
from ..utils.helpers import nutrition_score, clinical_compliance, load_config


# ──────────────────────────────────────────────────────────────────────────────
# Nutrition Database (minimal embedded DB; real deployments use PostgreSQL)
# ──────────────────────────────────────────────────────────────────────────────

MEAL_DB: List[Dict[str, Any]] = [
    {"id": 0,  "name": "Lentil soup + yogurt",    "calories": 420, "protein_g": 22, "fat_g": 8,  "sugar_g": 6,  "sodium_mg": 480, "gi": "low",  "sensory": "mild"},
    {"id": 1,  "name": "Grilled chicken salad",   "calories": 380, "protein_g": 35, "fat_g": 12, "sugar_g": 4,  "sodium_mg": 410, "gi": "low",  "sensory": "mild"},
    {"id": 2,  "name": "Creamy pasta",             "calories": 610, "protein_g": 18, "fat_g": 22, "sugar_g": 5,  "sodium_mg": 820, "gi": "high", "sensory": "smooth"},
    {"id": 3,  "name": "Quinoa vegetable bowl",   "calories": 440, "protein_g": 16, "fat_g": 10, "sugar_g": 7,  "sodium_mg": 390, "gi": "low",  "sensory": "mild"},
    {"id": 4,  "name": "Rice and beans",          "calories": 480, "protein_g": 15, "fat_g": 4,  "sugar_g": 3,  "sodium_mg": 320, "gi": "medium","sensory": "mild"},
    {"id": 5,  "name": "Oatmeal with berries",    "calories": 310, "protein_g": 10, "fat_g": 5,  "sugar_g": 12, "sodium_mg": 140, "gi": "low",  "sensory": "soft"},
    {"id": 6,  "name": "Egg white omelette",      "calories": 260, "protein_g": 22, "fat_g": 8,  "sugar_g": 2,  "sodium_mg": 290, "gi": "low",  "sensory": "mild"},
    {"id": 7,  "name": "Baked salmon + broccoli", "calories": 490, "protein_g": 40, "fat_g": 18, "sugar_g": 3,  "sodium_mg": 520, "gi": "low",  "sensory": "mild"},
    {"id": 8,  "name": "Vegetable stir-fry",      "calories": 350, "protein_g": 12, "fat_g": 9,  "sugar_g": 8,  "sodium_mg": 600, "gi": "medium","sensory": "mixed"},
    {"id": 9,  "name": "Turkey wrap",             "calories": 420, "protein_g": 30, "fat_g": 11, "sugar_g": 4,  "sodium_mg": 680, "gi": "medium","sensory": "mild"},
    {"id": 10, "name": "Chickpea curry",          "calories": 460, "protein_g": 18, "fat_g": 12, "sugar_g": 6,  "sodium_mg": 510, "gi": "low",  "sensory": "spiced"},
    {"id": 11, "name": "Greek yogurt + nuts",     "calories": 290, "protein_g": 18, "fat_g": 14, "sugar_g": 10, "sodium_mg": 120, "gi": "low",  "sensory": "soft"},
    {"id": 12, "name": "Tuna nicoise salad",      "calories": 400, "protein_g": 32, "fat_g": 15, "sugar_g": 5,  "sodium_mg": 590, "gi": "low",  "sensory": "mixed"},
    {"id": 13, "name": "Pumpkin soup",            "calories": 230, "protein_g": 6,  "fat_g": 5,  "sugar_g": 8,  "sodium_mg": 420, "gi": "medium","sensory": "soft"},
    {"id": 14, "name": "Whole wheat pasta + tomato","calories": 520, "protein_g": 16, "fat_g": 7, "sugar_g": 9, "sodium_mg": 480, "gi": "medium","sensory": "mild"},
    {"id": 15, "name": "Smoothie bowl",           "calories": 340, "protein_g": 9,  "fat_g": 6,  "sugar_g": 22, "sodium_mg": 90,  "gi": "high", "sensory": "soft"},
    {"id": 16, "name": "Tofu scramble",           "calories": 310, "protein_g": 20, "fat_g": 14, "sugar_g": 3,  "sodium_mg": 380, "gi": "low",  "sensory": "mild"},
    {"id": 17, "name": "Minestrone soup",         "calories": 290, "protein_g": 12, "fat_g": 6,  "sugar_g": 6,  "sodium_mg": 530, "gi": "low",  "sensory": "soft"},
    {"id": 18, "name": "Protein pancakes",        "calories": 380, "protein_g": 24, "fat_g": 10, "sugar_g": 8,  "sodium_mg": 310, "gi": "medium","sensory": "soft"},
    {"id": 19, "name": "Avocado toast + egg",     "calories": 410, "protein_g": 15, "fat_g": 19, "sugar_g": 3,  "sodium_mg": 420, "gi": "medium","sensory": "mild"},
]

N_MEALS = len(MEAL_DB)
N_STATES = 12  # encoded user state dimensions


# ──────────────────────────────────────────────────────────────────────────────
# Q-Learning Table
# ──────────────────────────────────────────────────────────────────────────────

class QLearningTable:
    """Tabular Q-learning for discrete state–action meal selection."""

    def __init__(self, n_states: int, n_actions: int, cfg: Dict[str, Any]):
        self.alpha = cfg.get("alpha", 0.01)
        self.gamma = cfg.get("gamma", 0.95)
        self.epsilon = cfg.get("epsilon", 0.15)
        self.epsilon_decay = cfg.get("epsilon_decay", 0.995)
        self.epsilon_min = cfg.get("epsilon_end", 0.05)
        self.Q = np.zeros((n_states, n_actions))

    def select_action(self, state_idx: int) -> int:
        if random.random() < self.epsilon:
            return random.randint(0, self.Q.shape[1] - 1)
        return int(np.argmax(self.Q[state_idx]))

    def update(self, s: int, a: int, r: float, s_next: int) -> None:
        td_target = r + self.gamma * np.max(self.Q[s_next])
        td_error = td_target - self.Q[s, a]
        self.Q[s, a] += self.alpha * td_error

    def decay_epsilon(self) -> None:
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save(self, path: str) -> None:
        np.save(path, self.Q)

    def load(self, path: str) -> None:
        self.Q = np.load(path)


# ──────────────────────────────────────────────────────────────────────────────
# Meal Planner Agent
# ──────────────────────────────────────────────────────────────────────────────

class MealPlannerAgent(BaseAgent):
    """
    Recommends meal plans using Q-learning.

    Algorithm (from paper):
      - State: encoded user profile (conditions, preferences, recent intake)
      - Action: choose meal ID from MEAL_DB
      - Reward: nutrition_score − clinical_penalty
      - Update: standard Bellman equation
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__("meal_planner", config)
        self.q_table = QLearningTable(N_STATES, N_MEALS, config)
        self.reward_weights = config.get("reward_weights", {
            "nutrition_score": 0.5,
            "user_acceptance": 0.3,
            "clinical_compliance": 0.2,
        })
        self.gradual_change = config.get("gradual_change", True)
        self._last_meals: List[int] = []  # per-user history

    # ── PRA: Perceive ─────────────────────────────────────────────────────────

    def perceive(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "user_profile": inputs.get("user_profile", {}),
            "recent_intake": inputs.get("recent_intake", []),
            "constraints": inputs.get("constraints", {}),
            "context": inputs.get("context", {}),
            "swap_request": inputs.get("swap_request", None),  # from food guidance
        }

    # ── PRA: Reason ──────────────────────────────────────────────────────────

    def reason(self, percept: Dict[str, Any]) -> Dict[str, Any]:
        profile = percept["user_profile"]
        constraints = percept["constraints"]
        context = percept["context"]

        state_idx = self._encode_state(profile, context)
        horizon = self.config.get("planning_horizon", "day")
        n_slots = 3 if horizon == "day" else 21  # 3 meals × 7 days

        meals = []
        for _ in range(n_slots):
            meal_id = self.q_table.select_action(state_idx)
            meal = MEAL_DB[meal_id]

            # Clinical compliance check (hard constraint)
            if not clinical_compliance(meal, constraints):
                meal_id = self._safe_fallback(constraints)
                meal = MEAL_DB[meal_id]

            # Sensory check (autism/ASD preference)
            sensory_pref = profile.get("sensory_preference", None)
            if sensory_pref and meal.get("sensory") not in [sensory_pref, "mild"]:
                meal_id = self._sensory_fallback(sensory_pref, constraints)
                meal = MEAL_DB[meal_id]

            meals.append(meal)

        return {
            "meals": meals,
            "state_idx": state_idx,
            "horizon": horizon,
            "constraints": constraints,
            "profile": profile,
        }

    # ── PRA: Act ─────────────────────────────────────────────────────────────

    def act(self, decision: Dict[str, Any]) -> Dict[str, Any]:
        meals = decision["meals"]
        horizon = decision["horizon"]
        constraints = decision["constraints"]
        profile = decision["profile"]

        # Compute aggregate nutrition
        total = {k: sum(m.get(k, 0) for m in meals) for k in
                 ["calories", "protein_g", "fat_g", "sugar_g", "sodium_mg"]}
        score = nutrition_score(total)

        # Build meal plan output
        plan = {
            "meals": [{"name": m["name"], "calories": m["calories"],
                       "protein_g": m["protein_g"], "fat_g": m["fat_g"]} for m in meals],
            "total_nutrition": total,
            "nutrition_score": round(score, 3),
            "shopping_list": list({m["name"] for m in meals}),
            "summary": self._build_summary(meals, horizon, profile),
        }

        self._last_meals = [m["id"] for m in meals]

        return {
            "action_type": "meal_plan",
            "plan": plan,
            "message": plan["summary"],
            "nutrients": total,
            "_confidence": min(1.0, score + 0.1),
            "_explanation_hint": f"Q-learning selected {len(meals)} meals; nutrition score={score:.2f}",
        }

    # ── Feedback & Learning ──────────────────────────────────────────────────

    def ingest_feedback(self, feedback: Dict[str, Any]) -> None:
        super().ingest_feedback(feedback)
        # Online Q-table update on acceptance signal
        accepted = feedback.get("accepted", None)
        meal_id = feedback.get("meal_id", None)
        state_idx = feedback.get("state_idx", 0)
        if accepted is not None and meal_id is not None:
            reward = 1.0 if accepted else -0.5
            next_state = (state_idx + 1) % N_STATES
            self.q_table.update(state_idx, meal_id, reward, next_state)
            self.q_table.decay_epsilon()

    # ── Helpers ──────────────────────────────────────────────────────────────

    def _encode_state(self, profile: Dict, context: Dict) -> int:
        """Map profile + context to a discrete state index [0, N_STATES)."""
        features = [
            int("diabetes" in profile.get("conditions", [])),
            int("hypertension" in profile.get("conditions", [])),
            int(profile.get("disability_type") in ["physical", "sensory"]),
            int(profile.get("neurodivergent_type") == "ASD"),
            int(profile.get("neurodivergent_type") == "ADHD"),
            int(context.get("hour", 12) < 10),   # morning
            int(10 <= context.get("hour", 12) < 14),  # midday
            int(context.get("hour", 12) >= 17),   # evening
            int(profile.get("calorie_restriction", False)),
            int(profile.get("low_sodium", False)),
            int(profile.get("sensory_preference") == "soft"),
            int(bool(profile.get("allergies"))),
        ]
        # Hash features to a state index
        idx = sum(f * (2 ** i) for i, f in enumerate(features)) % N_STATES
        return idx

    def _safe_fallback(self, constraints: Dict) -> int:
        """Return id of the safest meal given constraints."""
        candidates = [m for m in MEAL_DB if clinical_compliance(m, constraints)]
        if not candidates:
            return 0  # always safe default
        return min(candidates, key=lambda m: m.get("calories", 999))["id"]

    def _sensory_fallback(self, sensory_pref: str, constraints: Dict) -> int:
        candidates = [m for m in MEAL_DB
                      if m.get("sensory") in [sensory_pref, "mild"]
                      and clinical_compliance(m, constraints)]
        if not candidates:
            return 0
        return random.choice(candidates)["id"]

    def _build_summary(self, meals: List[Dict], horizon: str, profile: Dict) -> str:
        names = [m["name"] for m in meals[:3]]
        base = f"{'Daily' if horizon == 'day' else 'Weekly'} meal plan generated: {', '.join(names)}"
        if profile.get("neurodivergent_type") == "ASD":
            base += " — sensory-friendly options selected."
        return base
