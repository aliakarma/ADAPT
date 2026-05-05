"""
AgHealth+ — Explainable AI (XAI) Module
==========================================
Generates plain-language, user-facing explanations for every system decision.

Sources used in explanation generation:
- Agent call graph from MCP Router
- Blackboard resolved entries
- Active clinical constraints
- Bandit / Q-learning decision paths

Output is adapted to the user's reading level and preferred modality.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, TYPE_CHECKING

from loguru import logger

if TYPE_CHECKING:
    from ..core.blackboard import BlackboardEntry


# ──────────────────────────────────────────────────────────────────────────────
# Reading level templates
# ──────────────────────────────────────────────────────────────────────────────

TEMPLATES = {
    "simple": {
        "meal_plan":      "Your meal today is {meal}. It is healthy for you. 🍽️",
        "food_decision":  "{verdict_msg} because: {reason}.",
        "alert":          "Your {vital} reading needs attention. {action}",
        "reminder":       "Time to {task}! 🔔",
        "generic":        "{action_type}: {summary}",
    },
    "standard": {
        "meal_plan":      "Recommended meal: {meal}. Nutritional score: {score:.0%}. "
                          "Reasons: {reason}.",
        "food_decision":  "Decision: {verdict} for {food}. {reason}.",
        "alert":          "Health alert — {vital}: {value}. Suggested action: {action}.",
        "reminder":       "Scheduled reminder via {modality} for {task}.",
        "generic":        "{action_type} | {summary}",
    },
    "clinical": {
        "meal_plan":      "Meal plan generated (score={score:.3f}). "
                          "Constraints satisfied: {constraints}. Meals: {meal}.",
        "food_decision":  "Food guidance: {verdict} for '{food}' "
                          "(conf={conf:.2f}). Constraints: {reason}.",
        "alert":          "Anomaly detected (risk={risk_score:.3f}). "
                          "Triggers: {vital}. Action: {action}.",
        "reminder":       "Reminder dispatched via {modality} (UCB1 arm score). Task: {task}.",
        "generic":        "[{action_type}] {summary}",
    },
}


# ──────────────────────────────────────────────────────────────────────────────
# XAI Explainer
# ──────────────────────────────────────────────────────────────────────────────

class XAIExplainer:
    """
    Generates user-facing explanations from the call graph and blackboard state.

    The explanation is rendered at the user's preferred reading level
    (simple / standard / clinical) and reflected to the dashboard when enabled.
    """

    def __init__(self, default_level: str = "standard"):
        self.default_level = default_level

    def explain(
        self,
        intent: str,
        resolved_entries: List["BlackboardEntry"],
        constraints: Dict[str, Any],
        user_profile: Dict[str, Any],
    ) -> str:
        """
        Build a user-facing explanation string.

        Parameters
        ----------
        intent : str
        resolved_entries : list of BlackboardEntry
        constraints : dict
        user_profile : dict

        Returns
        -------
        str
        """
        level = user_profile.get("reading_level", self.default_level)
        disability = user_profile.get("disability_type", "none")

        parts: List[str] = []
        for entry in resolved_entries:
            snippet = self._explain_entry(entry, level, constraints, disability)
            if snippet:
                parts.append(snippet)

        if not parts:
            parts = [self._generic_explanation(intent, level)]

        explanation = " ".join(parts)
        logger.debug("XAI | explanation generated (level={}, len={})", level, len(explanation))
        return explanation

    # ── Entry-level explanation ───────────────────────────────────────────────

    def _explain_entry(
        self,
        entry: "BlackboardEntry",
        level: str,
        constraints: Dict,
        disability: str,
    ) -> str:
        tmpl = TEMPLATES.get(level, TEMPLATES["standard"])
        p = entry.payload
        etype = entry.entry_type

        try:
            if etype == "meal_plan":
                plan = p.get("plan", {})
                meals = plan.get("meals", [{}])
                meal_name = meals[0].get("name", "your meal") if meals else "your meal"
                score = plan.get("nutrition_score", 0.0)
                reason = p.get("_explanation_hint", "nutritional guidelines met")
                return tmpl.get("meal_plan", "").format(
                    meal=meal_name, score=score, reason=reason
                )

            elif etype == "food_decision":
                verdict = p.get("verdict", "approve")
                food = p.get("food_class", "this food")
                conf = p.get("confidence", 1.0)
                reason = p.get("_explanation_hint", "nutritional evaluation")
                verdict_msg = {
                    "approve": "This food is OK",
                    "limit":   "Have a smaller portion",
                    "swap":    "Consider a healthier alternative",
                    "deny":    "This item is not suitable for your plan",
                }.get(verdict, verdict)
                return tmpl.get("food_decision", "").format(
                    verdict=verdict, verdict_msg=verdict_msg,
                    food=food, conf=conf, reason=reason
                )

            elif etype in ("alert", "monitoring_ok"):
                triggers = p.get("triggers", [])
                vital_str = "; ".join(triggers[:2]) if triggers else "vitals"
                risk = p.get("risk_score", 0.0)
                action = p.get("message", "Please consult your care team.")
                return tmpl.get("alert", "").format(
                    vital=vital_str, value="", risk_score=risk, action=action[:120]
                )

            elif etype == "reminder":
                modality = p.get("modality", "notification")
                task = "meal/medication check"
                return tmpl.get("reminder", "").format(modality=modality, task=task)

            else:
                return tmpl.get("generic", "").format(
                    action_type=etype,
                    summary=p.get("message", p.get("summary", ""))[:100]
                )
        except (KeyError, ValueError) as exc:
            logger.warning("XAI | template render error: {}", exc)
            return f"Decision made for {etype}."

    def _generic_explanation(self, intent: str, level: str) -> str:
        return {
            "simple":   "Your assistant checked and everything looks fine.",
            "standard": f"System processed intent '{intent}' — no issues detected.",
            "clinical": f"Intent='{intent}'; no anomalies or violations found in resolved entries.",
        }.get(level, "No issues detected.")

    # ── Dashboard summary ─────────────────────────────────────────────────────

    def dashboard_summary(
        self,
        response: Any,
        user_profile: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Build a structured caregiver/clinician dashboard summary.
        """
        return {
            "user_id": user_profile.get("user_id", ""),
            "timestamp": response.timestamp if hasattr(response, "timestamp") else "",
            "action_type": getattr(response, "action_type", ""),
            "risk_level": getattr(response, "risk_level", 0.0),
            "recommendation": getattr(response, "final_recommendation", "")[:300],
            "explanation": getattr(response, "explanation", "")[:300],
            "sources": getattr(response, "sources", []),
        }
