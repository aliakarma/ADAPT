"""
ADAPT — Food Guidance Agent
==================================
PRA agent for real-time food recognition and nutritional guidance.

Perception : image/barcode input or NL query, pantry IoT data
Reasoning  : CNN classification → portion estimation → plan comparison → macro evaluation
Action     : approve / limit / swap decision + cooking guidance
Feedback   : user confirms/corrects → update preference model
"""
from __future__ import annotations

import random
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from loguru import logger

from .base_agent import BaseAgent
from .meal_planner import MEAL_DB
from ..utils.helpers import clinical_compliance


# ──────────────────────────────────────────────────────────────────────────────
# Simulated CNN Classifier
# (Production: MobileNetV2 fine-tuned on food dataset — see models/cnn_food_classifier.py)
# ──────────────────────────────────────────────────────────────────────────────

FOOD_CLASSES = [
    "salad", "pasta", "rice_dish", "soup", "sandwich",
    "fruit_bowl", "protein_plate", "legume_dish", "dairy_product", "snack_item"
]

# Map class → representative nutritional profile (per 100g serving)
CLASS_NUTRITION = {
    "salad":         {"calories": 80,  "protein_g": 3,  "fat_g": 4,  "sugar_g": 3,  "sodium_mg": 120, "gi": "low"},
    "pasta":         {"calories": 220, "protein_g": 8,  "fat_g": 10, "sugar_g": 2,  "sodium_mg": 380, "gi": "high"},
    "rice_dish":     {"calories": 180, "protein_g": 4,  "fat_g": 2,  "sugar_g": 1,  "sodium_mg": 200, "gi": "medium"},
    "soup":          {"calories": 110, "protein_g": 6,  "fat_g": 3,  "sugar_g": 3,  "sodium_mg": 450, "gi": "low"},
    "sandwich":      {"calories": 260, "protein_g": 12, "fat_g": 9,  "sugar_g": 4,  "sodium_mg": 520, "gi": "medium"},
    "fruit_bowl":    {"calories": 90,  "protein_g": 1,  "fat_g": 0,  "sugar_g": 18, "sodium_mg": 10,  "gi": "medium"},
    "protein_plate": {"calories": 240, "protein_g": 30, "fat_g": 10, "sugar_g": 1,  "sodium_mg": 360, "gi": "low"},
    "legume_dish":   {"calories": 160, "protein_g": 10, "fat_g": 3,  "sugar_g": 2,  "sodium_mg": 290, "gi": "low"},
    "dairy_product": {"calories": 120, "protein_g": 8,  "fat_g": 6,  "sugar_g": 10, "sodium_mg": 100, "gi": "low"},
    "snack_item":    {"calories": 200, "protein_g": 3,  "fat_g": 11, "sugar_g": 12, "sodium_mg": 220, "gi": "high"},
}

# Simulated per-class accuracy (matching paper's 99% reported result)
SIM_ACCURACY = {cls: 0.99 for cls in FOOD_CLASSES}


class SimulatedCNN:
    """
    Lightweight simulation of a MobileNetV2 food classifier.
    Returns (label, confidence) from a simulated probability distribution.
    In production, replace with `models/cnn_food_classifier.py`.
    """

    def predict(self, image_hint: str = "") -> Tuple[str, float]:
        """
        Parameters
        ----------
        image_hint : str
            Optional text hint to bias classification (for testing/simulation).
        """
        if image_hint:
            hint_lower = image_hint.lower()
            for cls in FOOD_CLASSES:
                if cls.replace("_", " ") in hint_lower or hint_lower in cls:
                    conf = np.random.uniform(0.92, 0.99)
                    return cls, round(conf, 3)

        # Random class with high-confidence distribution
        cls = random.choice(FOOD_CLASSES)
        conf = np.random.beta(a=15, b=1)  # heavily skewed towards high confidence
        return cls, round(float(conf), 3)

    def estimate_portion_g(self, cls: str) -> float:
        """Estimate portion size in grams (simplified heuristic)."""
        portion_map = {
            "pasta": 350, "rice_dish": 280, "salad": 200, "soup": 300,
            "sandwich": 180, "fruit_bowl": 250, "protein_plate": 200,
            "legume_dish": 280, "dairy_product": 150, "snack_item": 50,
        }
        base = portion_map.get(cls, 200)
        noise = np.random.normal(0, 20)
        return max(50.0, round(base + noise, 1))


# ──────────────────────────────────────────────────────────────────────────────
# Food Guidance Agent
# ──────────────────────────────────────────────────────────────────────────────

class FoodGuidanceAgent(BaseAgent):
    """
    Identifies food, estimates nutrients, and compares with clinical plan.

    Returns: approve / limit / swap + (optional) cooking guidance steps.
    """

    VERDICTS = {
        "approve": "✅ This looks good for your plan.",
        "limit":   "⚠️  You can have a smaller portion of this.",
        "swap":    "🔄 Consider a healthier swap — see alternatives below.",
        "deny":    "❌ This item conflicts with your dietary restrictions.",
    }

    def __init__(self, config: Dict[str, Any]):
        super().__init__("food_guidance", config)
        self.cnn = SimulatedCNN()
        self.conf_threshold = config.get("confidence_threshold", 0.75)
        self.thresholds = config.get("approve_limit_deny_thresholds", {
            "approve": 1.0, "limit": 1.25, "deny": 1.5
        })

    # ── PRA: Perceive ─────────────────────────────────────────────────────────

    def perceive(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        input_type = "image" if inputs.get("image") or inputs.get("image_hint") else "query"
        return {
            "input_type": input_type,
            "image_hint": inputs.get("image_hint", ""),
            "query": inputs.get("query", ""),
            "user_profile": inputs.get("user_profile", {}),
            "constraints": inputs.get("constraints", {}),
            "current_plan": inputs.get("current_plan", {}),
            "context": inputs.get("context", {}),
        }

    # ── PRA: Reason ──────────────────────────────────────────────────────────

    def reason(self, percept: Dict[str, Any]) -> Dict[str, Any]:
        constraints = percept["constraints"]
        profile = percept["user_profile"]

        if percept["input_type"] == "image":
            food_class, confidence = self.cnn.predict(percept["image_hint"])

            # Request clarification if confidence is too low
            if confidence < self.conf_threshold:
                return {
                    "needs_clarification": True,
                    "message": "I'm not sure what this food is. Could you take another photo or tell me what it is?",
                    "confidence": confidence,
                }

            portion_g = self.cnn.estimate_portion_g(food_class)
            base_nutrition = CLASS_NUTRITION.get(food_class, CLASS_NUTRITION["snack_item"])
            # Scale by portion
            scale = portion_g / 100.0
            nutrients = {k: round(v * scale, 1) for k, v in base_nutrition.items()
                         if k not in ["gi"]}
            nutrients["gi"] = base_nutrition.get("gi", "medium")

            # Compare with plan limits
            verdict = self._evaluate_verdict(food_class, nutrients, constraints, profile)
            alternatives = self._get_alternatives(constraints, profile) if verdict in ["swap", "limit"] else []

            return {
                "needs_clarification": False,
                "food_class": food_class,
                "portion_g": portion_g,
                "nutrients": nutrients,
                "verdict": verdict,
                "alternatives": alternatives,
                "confidence": confidence,
                "constraints": constraints,
            }
        else:
            # NLP path — parse query intent
            query = percept["query"].lower()
            food_class = self._classify_query(query)
            base_nutrition = CLASS_NUTRITION.get(food_class, CLASS_NUTRITION["snack_item"])
            nutrients = {k: v for k, v in base_nutrition.items() if k != "gi"}
            nutrients["gi"] = base_nutrition.get("gi", "medium")
            verdict = self._evaluate_verdict(food_class, nutrients, constraints, profile)
            alternatives = self._get_alternatives(constraints, profile) if verdict in ["swap", "limit"] else []
            return {
                "needs_clarification": False,
                "food_class": food_class,
                "portion_g": 100.0,
                "nutrients": nutrients,
                "verdict": verdict,
                "alternatives": alternatives,
                "confidence": 0.85,
                "constraints": constraints,
            }

    # ── PRA: Act ─────────────────────────────────────────────────────────────

    def act(self, decision: Dict[str, Any]) -> Dict[str, Any]:
        if decision.get("needs_clarification"):
            return {
                "action_type": "clarification_needed",
                "message": decision["message"],
                "_confidence": decision.get("confidence", 0.0),
                "_explanation_hint": "Low confidence food classification — requesting user clarification.",
            }

        verdict = decision["verdict"]
        food_class = decision["food_class"]
        nutrients = decision["nutrients"]
        alternatives = decision.get("alternatives", [])

        message = self.VERDICTS.get(verdict, "")
        cooking_steps = self._cooking_guidance(food_class, verdict) if verdict == "approve" else []

        return {
            "action_type": "food_decision",
            "verdict": verdict,
            "food_class": food_class,
            "portion_g": decision.get("portion_g", 100.0),
            "nutrients": nutrients,
            "message": message,
            "alternatives": alternatives,
            "cooking_steps": cooking_steps,
            "confidence": decision.get("confidence", 1.0),
            "_confidence": decision.get("confidence", 1.0),
            "_explanation_hint": (
                f"CNN identified '{food_class}' (conf={decision.get('confidence', 1):.2f}). "
                f"Verdict='{verdict}' based on {list(decision.get('constraints', {}).keys())}."
            ),
        }

    # ── Private helpers ───────────────────────────────────────────────────────

    def _evaluate_verdict(self, food_class: str, nutrients: Dict, constraints: Dict, profile: Dict) -> str:
        # Hard deny: allergies or forbidden items
        forbidden = profile.get("allergies", []) + constraints.get("forbidden_items", [])
        if any(f.lower() in food_class.lower() for f in forbidden):
            return "deny"

        # Check clinical compliance
        if not clinical_compliance(nutrients, constraints):
            return "swap"

        # Check GI preference (diabetes)
        gi_val = nutrients.get("gi", "medium")
        if constraints.get("low_gi_preferred") and gi_val == "high":
            return "limit"

        # Check caloric ratio vs plan
        plan_calories = profile.get("daily_calorie_target", 2000)
        meal_calories = nutrients.get("calories", 0)
        ratio = (meal_calories * 3) / plan_calories  # assume 3 meals/day
        if ratio > self.thresholds.get("deny", 1.5):
            return "swap"
        elif ratio > self.thresholds.get("limit", 1.25):
            return "limit"
        return "approve"

    def _get_alternatives(self, constraints: Dict, profile: Dict) -> List[Dict]:
        candidates = [
            m for m in MEAL_DB
            if clinical_compliance(m, constraints)
            and m.get("gi", "medium") != "high"
        ]
        random.shuffle(candidates)
        return [{"name": m["name"], "calories": m["calories"]} for m in candidates[:3]]

    @staticmethod
    def _classify_query(query: str) -> str:
        for cls in FOOD_CLASSES:
            if cls.replace("_", " ") in query:
                return cls
        if any(w in query for w in ["pasta", "noodle", "spaghetti"]):
            return "pasta"
        if any(w in query for w in ["rice", "fried rice"]):
            return "rice_dish"
        if any(w in query for w in ["salad", "lettuce"]):
            return "salad"
        return "snack_item"

    @staticmethod
    def _cooking_guidance(food_class: str, verdict: str) -> List[str]:
        guides = {
            "salad": ["1. Rinse vegetables.", "2. Chop into bite-sized pieces.", "3. Add low-sodium dressing."],
            "soup":  ["1. Heat broth on medium.", "2. Add vegetables.", "3. Simmer 10 min."],
            "legume_dish": ["1. Rinse lentils.", "2. Boil with water 15 min.", "3. Season lightly."],
        }
        return guides.get(food_class, ["Follow standard preparation for this item."])
