"""
AgHealth+ — Central Reasoning Core
=====================================
Coordinates all agents through the shared Blackboard.

Responsibilities
----------------
1. Read agent outputs posted to the blackboard.
2. Resolve conflicts using weighted priority rules:
     medical_safety (1.0) > user_preference (0.6) > behavioral_nudge (0.3)
3. Fetch supplementary data from the Data/Policy layer when needed.
4. Assemble a coherent final response from potentially multiple agent outputs.
5. Trigger downstream XAI to generate user-facing explanations.
6. Record a learning update signal for adaptive agents.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from loguru import logger

from .blackboard import Blackboard, BlackboardEntry, Priority
from ..utils.helpers import generate_trace_id, now_utc

if TYPE_CHECKING:
    from ..xai.explainer import XAIExplainer


# ──────────────────────────────────────────────────────────────────────────────
# Response container
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class CoordinatedResponse:
    trace_id: str
    final_recommendation: str
    action_type: str                     # "meal_plan", "alert", "reminder", etc.
    payload: Dict[str, Any] = field(default_factory=dict)
    explanation: str = ""
    caregiver_summary: Optional[str] = None
    risk_level: float = 0.0              # 0 = safe, 1 = critical
    sources: List[str] = field(default_factory=list)
    timestamp: str = field(default_factory=now_utc)


# ──────────────────────────────────────────────────────────────────────────────
# Reasoning Core
# ──────────────────────────────────────────────────────────────────────────────

class ReasoningCore:
    """
    Central coordinator that sits above all agents.

    Parameters
    ----------
    blackboard : Blackboard
    priority_weights : dict
        Weights per priority tier (optional override).
    xai_explainer : XAIExplainer | None
        If provided, explanations are generated for every response.
    """

    PRIORITY_WEIGHTS = {
        Priority.MEDICAL_SAFETY: 1.0,
        Priority.USER_PREFERENCE: 0.6,
        Priority.BEHAVIORAL_NUDGE: 0.3,
    }

    def __init__(
        self,
        blackboard: Blackboard,
        priority_weights: Optional[Dict] = None,
        xai_explainer: Optional["XAIExplainer"] = None,
    ):
        self.blackboard = blackboard
        self._weights = {
            p: priority_weights.get(p.name.lower(), w)
            for p, w in self.PRIORITY_WEIGHTS.items()
        } if priority_weights else self.PRIORITY_WEIGHTS
        self.xai = xai_explainer

    # ── Main coordination ────────────────────────────────────────────────────

    def coordinate(
        self,
        intent_plan: Dict[str, Any],
        agent_results: Dict[str, Dict[str, Any]],
        user_profile: Dict[str, Any],
        context: Dict[str, Any],
        trace_id: str = "",
    ) -> CoordinatedResponse:
        """
        Merge and reconcile outputs from one or more agents into a single
        user-facing response.

        Parameters
        ----------
        intent_plan : dict
            Output of LLMDecisionNode.decide()
        agent_results : dict
            {tool_name: ToolResult.payload}
        user_profile : dict
        context : dict
        trace_id : str
        """
        trace_id = trace_id or generate_trace_id()
        logger.info("ReasoningCore | coordinate | trace={} agents={}", trace_id[:8], list(agent_results.keys()))

        # 1. Post agent results to blackboard
        for agent_id, result in agent_results.items():
            priority = self._infer_priority(agent_id, result)
            self.blackboard.post(
                agent_id=agent_id,
                entry_type=result.get("action_type", "unknown"),
                payload=result,
                priority=priority,
                trace_id=trace_id,
            )

        # 2. Collect relevant entries from blackboard
        entries = self.blackboard.query(resolved=False, limit=20)

        # 3. Resolve conflicts
        resolved = self._resolve_conflicts(entries, intent_plan.get("constraints", {}))

        # 4. Build final recommendation
        recommendation, action_type, payload, risk_level = self._synthesise(resolved, agent_results, context)

        # 5. Generate explanation
        explanation = ""
        if self.xai:
            explanation = self.xai.explain(
                intent=intent_plan.get("intent", ""),
                resolved_entries=resolved,
                constraints=intent_plan.get("constraints", {}),
                user_profile=user_profile,
            )

        # 6. Caregiver summary (if high-risk or enabled)
        caregiver_summary = None
        if risk_level >= 0.7 or context.get("caregiver_updates_enabled", False):
            caregiver_summary = self._build_caregiver_summary(recommendation, risk_level, payload)

        # 7. Mark entries resolved
        for e in entries:
            self.blackboard.mark_resolved(e)

        # 8. Store response in blackboard context for learning update
        self.blackboard.set_context("last_response", {
            "recommendation": recommendation,
            "risk_level": risk_level,
            "trace_id": trace_id,
        })

        response = CoordinatedResponse(
            trace_id=trace_id,
            final_recommendation=recommendation,
            action_type=action_type,
            payload=payload,
            explanation=explanation,
            caregiver_summary=caregiver_summary,
            risk_level=risk_level,
            sources=list(agent_results.keys()),
        )
        logger.info("ReasoningCore | done | action={} risk={:.2f}", action_type, risk_level)
        return response

    # ── Conflict resolution ──────────────────────────────────────────────────

    def _resolve_conflicts(
        self,
        entries: List[BlackboardEntry],
        constraints: Dict[str, Any],
    ) -> List[BlackboardEntry]:
        """
        Apply weighted priority resolution.
        If a MEDICAL_SAFETY entry contradicts a lower-priority entry on the
        same topic, the lower-priority entry is discarded.
        """
        # Group by topic (entry_type)
        by_type: Dict[str, List[BlackboardEntry]] = {}
        for e in entries:
            by_type.setdefault(e.entry_type, []).append(e)

        resolved: List[BlackboardEntry] = []
        for etype, group in by_type.items():
            # Sort by priority descending; keep highest
            group.sort(key=lambda x: -x.priority)
            winner = group[0]

            # Extra: if medical safety is in the group, ensure constraints are honoured
            if winner.priority == Priority.MEDICAL_SAFETY:
                winner = self._enforce_constraints(winner, constraints)

            resolved.append(winner)
            for loser in group[1:]:
                logger.debug(
                    "ReasoningCore | conflict | kept={} (p={}) discarded={} (p={}) type={}",
                    winner.agent_id, winner.priority, loser.agent_id, loser.priority, etype
                )

        return resolved

    def _enforce_constraints(
        self,
        entry: BlackboardEntry,
        constraints: Dict[str, Any],
    ) -> BlackboardEntry:
        """Modify entry payload to enforce hard clinical constraints."""
        payload = dict(entry.payload)
        nutrients = payload.get("nutrients", {})
        for nutrient, rule in constraints.items():
            if isinstance(rule, dict):
                val = nutrients.get(nutrient, 0)
                if "max" in rule and val > rule["max"]:
                    logger.warning(
                        "ReasoningCore | constraint violation | {}={} > max={}",
                        nutrient, val, rule["max"]
                    )
                    payload["constraint_violation"] = True
                    payload["violated_constraint"] = nutrient
        entry.payload = payload
        return entry

    # ── Synthesis ────────────────────────────────────────────────────────────

    def _synthesise(
        self,
        resolved: List[BlackboardEntry],
        agent_results: Dict[str, Any],
        context: Dict[str, Any],
    ):
        """Produce final recommendation string, action type, payload, risk."""
        risk_level = 0.0
        action_type = "info"
        payload: Dict[str, Any] = {}
        parts: List[str] = []

        for e in resolved:
            p = e.payload
            if e.entry_type == "alert":
                risk_level = max(risk_level, float(p.get("risk_score", 0.5)))
                parts.append(p.get("message", "Health alert detected."))
                action_type = "alert"
                payload.update(p)
            elif e.entry_type == "meal_plan":
                parts.append(p.get("summary", "Meal plan updated."))
                action_type = "meal_plan"
                payload.update(p)
            elif e.entry_type == "food_decision":
                verdict = p.get("verdict", "approve")
                parts.append(p.get("message", f"Food decision: {verdict}."))
                action_type = "food_decision"
                payload.update(p)
            elif e.entry_type == "reminder":
                parts.append(p.get("message", "Reminder set."))
                action_type = action_type if action_type != "info" else "reminder"
                payload.update(p)

        recommendation = " ".join(parts) if parts else "All checks passed. No action required."
        return recommendation, action_type, payload, risk_level

    # ── Priority inference ───────────────────────────────────────────────────

    def _infer_priority(self, agent_id: str, result: Dict[str, Any]) -> Priority:
        if agent_id == "monitoring" and result.get("risk_score", 0) > 0.6:
            return Priority.MEDICAL_SAFETY
        if agent_id in ("meal_planner", "food_guidance"):
            if result.get("constraint_violation"):
                return Priority.MEDICAL_SAFETY
            return Priority.USER_PREFERENCE
        if agent_id == "reminder":
            return Priority.BEHAVIORAL_NUDGE
        return Priority.USER_PREFERENCE

    # ── Caregiver summary ────────────────────────────────────────────────────

    def _build_caregiver_summary(
        self,
        recommendation: str,
        risk_level: float,
        payload: Dict[str, Any],
    ) -> str:
        risk_label = "LOW" if risk_level < 0.4 else ("MODERATE" if risk_level < 0.7 else "HIGH")
        return (
            f"[Caregiver Summary] Risk: {risk_label} ({risk_level:.2f}). "
            f"System recommendation: {recommendation[:200]}. "
            f"Details: {str(payload)[:300]}."
        )
