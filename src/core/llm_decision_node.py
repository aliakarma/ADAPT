"""
AgHealth+ — LLM Decision Node
================================
Converts a normalised multimodal prompt into an ordered agent call-graph.

In production this wraps an LLM API (e.g., GPT-4o, Claude).
In offline / simulation mode (use_mock=True) a deterministic rule-based
parser produces equivalent intent structures — enabling full reproducibility
without API keys.

Output schema
-------------
{
  "intent":      str,               # e.g. "check_meal"
  "primary_tool": str,              # first agent to call
  "fallback_tool": str | None,      # if primary fails or defers
  "call_graph":  List[str],         # ordered tool names
  "constraints": Dict[str, Any],    # forwarded from user profile
  "rationale":   str,               # short plain-language reason
}
"""
import re
from typing import Any, Dict, List, Optional

from loguru import logger


# ──────────────────────────────────────────────────────────────────────────────
# Intent taxonomy
# ──────────────────────────────────────────────────────────────────────────────

INTENT_PATTERNS = [
    (re.compile(r"\b(check|scan|is this ok|what is|identify)\b.*\b(food|meal|eat|lunch|dinner|breakfast|snack)\b", re.I), "check_meal"),
    (re.compile(r"\b(plan|suggest|recommend)\b.*\b(meal|menu|diet|food)\b", re.I), "plan_meal"),
    (re.compile(r"\b(remind|alert|schedule|notify)\b", re.I), "set_reminder"),
    (re.compile(r"\b(monitor|vitals|heart|glucose|steps|sleep|health)\b", re.I), "check_health"),
    (re.compile(r"\b(help|assist|support)\b", re.I), "general_help"),
]

# Maps intent → primary tool → fallback tool
ROUTING_TABLE: Dict[str, Dict[str, Optional[str]]] = {
    "check_meal":   {"primary": "food_guidance", "fallback": "meal_planner"},
    "plan_meal":    {"primary": "meal_planner",  "fallback": None},
    "set_reminder": {"primary": "reminder",      "fallback": None},
    "check_health": {"primary": "monitoring",    "fallback": None},
    "general_help": {"primary": "meal_planner",  "fallback": "food_guidance"},
}


# ──────────────────────────────────────────────────────────────────────────────
# LLM Decision Node
# ──────────────────────────────────────────────────────────────────────────────

class LLMDecisionNode:
    """
    Parses structured prompts → intent → agent call-graph.

    Parameters
    ----------
    use_mock : bool
        If True, uses rule-based parsing (fully offline).
        If False, calls the configured LLM API.
    llm_config : dict
        Config dict from system_config.yaml `llm` section.
    """

    def __init__(self, use_mock: bool = True, llm_config: Optional[Dict] = None):
        self.use_mock = use_mock
        self.llm_config = llm_config or {}
        logger.info("LLMDecisionNode | mode={}", "mock" if use_mock else "api")

    # ── Public API ───────────────────────────────────────────────────────────

    def decide(
        self,
        prompt: str,
        user_profile: Dict[str, Any],
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Main entry point. Returns an intent plan dict.

        Parameters
        ----------
        prompt : str
            Normalised text from the Multimodal UI.
        user_profile : dict
            Contains clinical constraints, preferences, disability type, etc.
        context : dict
            Time, location, recent compliance, access mode, etc.
        """
        if self.use_mock:
            plan = self._mock_decide(prompt, user_profile, context)
        else:
            plan = self._api_decide(prompt, user_profile, context)

        logger.info(
            "LLMDecisionNode | intent='{}' primary='{}' fallback='{}'",
            plan["intent"], plan["primary_tool"], plan["fallback_tool"]
        )
        return plan

    # ── Mock (rule-based) ────────────────────────────────────────────────────

    def _mock_decide(
        self,
        prompt: str,
        user_profile: Dict[str, Any],
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        intent = self._classify_intent(prompt)
        routing = ROUTING_TABLE.get(intent, ROUTING_TABLE["general_help"])

        call_graph = [routing["primary"]]
        if routing["fallback"]:
            call_graph.append(routing["fallback"])

        # Build a constraints dict from user profile clinical rules
        constraints = self._extract_constraints(user_profile)

        rationale = self._build_rationale(intent, routing, constraints, context)

        return {
            "intent": intent,
            "primary_tool": routing["primary"],
            "fallback_tool": routing["fallback"],
            "call_graph": call_graph,
            "constraints": constraints,
            "rationale": rationale,
            "prompt_used": prompt[:120],
        }

    def _classify_intent(self, prompt: str) -> str:
        for pattern, intent in INTENT_PATTERNS:
            if pattern.search(prompt):
                return intent
        return "general_help"

    def _extract_constraints(self, profile: Dict[str, Any]) -> Dict[str, Any]:
        """Pull hard clinical constraints from user profile."""
        constraints: Dict[str, Any] = {}
        conditions = profile.get("conditions", [])
        if "diabetes" in conditions:
            constraints["glucose_limit"] = 180
            constraints["sugar_g"] = {"max": 30}
            constraints["low_gi_preferred"] = True
        if "hypertension" in conditions:
            constraints["sodium_mg"] = {"max": 1500}
        allergies = profile.get("allergies", [])
        if allergies:
            constraints["forbidden_items"] = allergies
        return constraints

    def _build_rationale(
        self,
        intent: str,
        routing: Dict,
        constraints: Dict,
        context: Dict,
    ) -> str:
        parts = [f"Intent classified as '{intent}'."]
        parts.append(f"Routing to '{routing['primary']}'.")
        if routing["fallback"]:
            parts.append(f"Fallback: '{routing['fallback']}' if primary defers.")
        if constraints:
            parts.append(f"Clinical constraints active: {list(constraints.keys())}.")
        hour = context.get("hour", None)
        if hour is not None:
            parts.append(f"Current time context: {hour:02d}:xx.")
        return " ".join(parts)

    # ── API mode (production) ────────────────────────────────────────────────

    def _api_decide(
        self,
        prompt: str,
        user_profile: Dict[str, Any],
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Calls a real LLM API.  Requires OPENAI_API_KEY or equivalent
        environment variable to be set.  Returns the same dict structure as
        _mock_decide for full system compatibility.
        """
        try:
            import openai, os, json as _json
            client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])
            system_prompt = self.llm_config.get("system_prompt", "")
            user_msg = (
                f"User prompt: {prompt}\n"
                f"Profile summary: conditions={user_profile.get('conditions')}, "
                f"disability={user_profile.get('disability_type')}\n"
                f"Context: {context}\n\n"
                "Return ONLY a JSON object with keys: intent, primary_tool, "
                "fallback_tool, call_graph, constraints, rationale."
            )
            response = client.chat.completions.create(
                model=self.llm_config.get("model", "gpt-4o-mini"),
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_msg},
                ],
                temperature=self.llm_config.get("temperature", 0.2),
                max_tokens=self.llm_config.get("max_tokens", 512),
            )
            raw = response.choices[0].message.content.strip()
            # Strip markdown fences if present
            raw = re.sub(r"```json|```", "", raw).strip()
            return _json.loads(raw)
        except Exception as exc:
            logger.warning("LLMDecisionNode | API call failed ({}), falling back to mock", exc)
            return self._mock_decide(prompt, user_profile, context)
