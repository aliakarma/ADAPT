"""
AgHealth+ — System Orchestrator
=================================
Ties together all layers into a single, end-to-end request-processing pipeline:

  User Input
      ↓
  Multimodal UI (context normalisation)
      ↓
  LLM Decision Node (intent → call graph)
      ↓
  MCP Router (dispatch to agents)
      ↓
  Agent PRA Loops (meal_planner / reminder / food_guidance / monitoring)
      ↓
  Central Reasoning Core + Blackboard (conflict resolution)
      ↓
  XAI Explainer (explanation generation)
      ↓
  Policy Store (data governance)
      ↓
  Coordinated Response → User + (optional) Caregiver Dashboard
"""
from __future__ import annotations

import asyncio
from datetime import datetime
from typing import Any, Dict, List, Optional

from loguru import logger

from .agents import MealPlannerAgent, ReminderAgent, FoodGuidanceAgent, MonitoringAgent
from .core import Blackboard, MCPRouter, LLMDecisionNode, ReasoningCore, ToolCall
from .core.blackboard import Priority
from .xai import XAIExplainer
from .policy import PolicyStore
from .utils import setup_logging, load_config, generate_trace_id, set_global_seed


# ──────────────────────────────────────────────────────────────────────────────
# Orchestrator
# ──────────────────────────────────────────────────────────────────────────────

class AgHealthOrchestrator:
    """
    Central coordinator for the AgHealth+ agentic system.

    Initialises all components from config files and exposes a single
    `process_request()` async method that handles the full PRA pipeline.
    """

    def __init__(
        self,
        system_cfg: Optional[Dict] = None,
        agent_cfg: Optional[Dict] = None,
        model_cfg: Optional[Dict] = None,
        seed: int = 42,
    ):
        set_global_seed(seed)
        sys_cfg = system_cfg or load_config("system_config")
        agt_cfg = agent_cfg or load_config("agent_configs")

        log_cfg = sys_cfg.get("system", {})
        setup_logging(
            log_dir=log_cfg.get("log_dir", "results/logs"),
            level=log_cfg.get("log_level", "INFO"),
        )

        logger.info("AgHealth+ | initialising orchestrator...")

        # ── Policy store ──────────────────────────────────────────────────────
        self.policy = PolicyStore(sys_cfg.get("policy", {}))

        # ── XAI ──────────────────────────────────────────────────────────────
        self.xai = XAIExplainer(default_level="standard")

        # ── Blackboard ────────────────────────────────────────────────────────
        bb_cfg = sys_cfg.get("reasoning", {})
        self.blackboard = Blackboard(
            max_entries=bb_cfg.get("max_blackboard_entries", 500),
            ttl_seconds=bb_cfg.get("blackboard_ttl_seconds", 3600),
        )

        # ── Agents ────────────────────────────────────────────────────────────
        self.meal_planner = MealPlannerAgent(agt_cfg.get("meal_planner", {}))
        self.reminder = ReminderAgent(agt_cfg.get("reminder", {}))
        self.food_guidance = FoodGuidanceAgent(agt_cfg.get("food_guidance", {}))
        self.monitoring = MonitoringAgent(agt_cfg.get("monitoring", {}))

        self._agents = {
            "meal_planner": self.meal_planner,
            "reminder":     self.reminder,
            "food_guidance": self.food_guidance,
            "monitoring":   self.monitoring,
        }
        for agent in self._agents.values():
            agent.start()

        # ── MCP Router ────────────────────────────────────────────────────────
        mcp_cfg = sys_cfg.get("mcp", {})
        self.mcp = MCPRouter(policy_store=None, timeout=mcp_cfg.get("timeout_seconds", 10))
        for name, agent in self._agents.items():
            self.mcp.register(name, self._make_handler(agent))

        # ── LLM Decision Node ─────────────────────────────────────────────────
        llm_cfg = sys_cfg.get("llm", {})
        self.llm = LLMDecisionNode(
            use_mock=llm_cfg.get("use_mock", True),
            llm_config=llm_cfg,
        )

        # ── Reasoning Core ────────────────────────────────────────────────────
        pw = bb_cfg.get("priority_weights", {})
        self.reasoning_core = ReasoningCore(
            blackboard=self.blackboard,
            priority_weights=pw,
            xai_explainer=self.xai,
        )

        logger.info("AgHealth+ | orchestrator ready | agents={}", list(self._agents.keys()))

    # ── Request processing ────────────────────────────────────────────────────

    async def process_request(
        self,
        prompt: str,
        user_profile: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None,
        modality_inputs: Optional[Dict[str, Any]] = None,
        trace_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Full end-to-end request pipeline (sequence diagram steps 1–19).

        Parameters
        ----------
        prompt : str
            Normalised user text/voice input.
        user_profile : dict
        context : dict
            {hour, location, recent_compliance, accessibility_mode, ...}
        modality_inputs : dict
            Extra inputs: {image, image_hint, vitals, intake_log, ...}
        trace_id : str

        Returns
        -------
        dict
            Full response with recommendation, explanation, caregiver_summary.
        """
        trace_id = trace_id or generate_trace_id()
        context = context or {"hour": datetime.now().hour}
        modality_inputs = modality_inputs or {}

        logger.info("Orchestrator | request | trace={} prompt='{}'",
                    trace_id[:8], prompt[:60])

        # Step 1 — Context normalisation
        context = self._normalise_context(context, user_profile)

        # Step 2 — LLM intent parsing
        intent_plan = self.llm.decide(prompt, user_profile, context)

        # Step 3 — Build agent inputs
        agent_inputs = self._build_agent_inputs(
            user_profile=user_profile,
            context=context,
            constraints=intent_plan.get("constraints", {}),
            modality_inputs=modality_inputs,
        )

        # Step 4 — MCP routing: run call graph
        call_graph = intent_plan.get("call_graph", [intent_plan.get("primary_tool")])
        agent_results: Dict[str, Dict] = {}

        for tool_name in call_graph:
            tool_call = ToolCall(
                tool_name=tool_name,
                intent=intent_plan["intent"],
                user_profile=user_profile,
                context=context,
                payload=agent_inputs,
                scope=["nutrition", "reminders", "health_vitals"],
                trace_id=trace_id,
            )
            result = await self.mcp.route(tool_call)
            if result.success:
                agent_results[tool_name] = result.payload

        # Step 5 — Reasoning Core: coordinate and resolve
        response = self.reasoning_core.coordinate(
            intent_plan=intent_plan,
            agent_results=agent_results,
            user_profile=user_profile,
            context=context,
            trace_id=trace_id,
        )

        # Step 6 — Build final output
        output = {
            "trace_id": trace_id,
            "intent": intent_plan["intent"],
            "recommendation": response.final_recommendation,
            "action_type": response.action_type,
            "explanation": response.explanation,
            "risk_level": response.risk_level,
            "caregiver_summary": response.caregiver_summary,
            "sources": response.sources,
            "rationale": intent_plan.get("rationale", ""),
            "payload": response.payload,
            "timestamp": response.timestamp,
        }

        logger.info("Orchestrator | done | action={} risk={:.2f} trace={}",
                    response.action_type, response.risk_level, trace_id[:8])
        return output

    # ── Sync wrapper ──────────────────────────────────────────────────────────

    def process_request_sync(self, **kwargs) -> Dict[str, Any]:
        """Synchronous wrapper for non-async contexts."""
        return asyncio.run(self.process_request(**kwargs))

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _normalise_context(context: Dict, profile: Dict) -> Dict:
        defaults = {
            "hour": datetime.now().hour,
            "location": "home",
            "accessibility_mode": profile.get("disability_type", "none"),
            "caregiver_updates_enabled": profile.get("caregiver_enabled", False),
            "recent_compliance": 0.7,
        }
        return {**defaults, **context}

    @staticmethod
    def _build_agent_inputs(
        user_profile: Dict,
        context: Dict,
        constraints: Dict,
        modality_inputs: Dict,
    ) -> Dict[str, Any]:
        return {
            "user_profile": user_profile,
            "context": context,
            "constraints": constraints,
            **modality_inputs,
        }

    @staticmethod
    def _make_handler(agent):
        """Wrap agent.run() as an async-compatible handler for MCP."""
        def handler(tool_call: ToolCall) -> Dict[str, Any]:
            result = agent.run(tool_call.payload, trace_id=tool_call.trace_id)
            return result.to_dict()
        return handler

    # ── Feedback ──────────────────────────────────────────────────────────────

    def send_feedback(
        self,
        agent_id: str,
        feedback: Dict[str, Any],
    ) -> None:
        """Forward user/caregiver feedback to the relevant agent."""
        agent = self._agents.get(agent_id)
        if agent:
            agent.ingest_feedback(feedback)
            logger.info("Orchestrator | feedback forwarded | agent={}", agent_id)
        else:
            logger.warning("Orchestrator | unknown agent for feedback: {}", agent_id)

    # ── Dashboard ─────────────────────────────────────────────────────────────

    def get_dashboard_snapshot(self) -> Dict[str, Any]:
        return {
            "blackboard": self.blackboard.snapshot(),
            "mcp_audit": self.mcp.audit_log()[-10:],
            "policy_summary": self.policy.audit_summary(),
        }
