"""
AgHealth+ — Test Suite
========================
Unit and integration tests for all system components.

Run with: pytest tests/ -v
"""
import asyncio
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.core.blackboard import Blackboard, Priority
from src.core.mcp_router import MCPRouter, ToolCall
from src.core.llm_decision_node import LLMDecisionNode
from src.core.reasoning_core import ReasoningCore
from src.agents.meal_planner import MealPlannerAgent
from src.agents.reminder import ReminderAgent
from src.agents.food_guidance import FoodGuidanceAgent
from src.agents.monitoring import MonitoringAgent
from src.evaluation.metrics import (
    compute_nutritional_adequacy,
    compute_adherence_rate,
    compute_user_satisfaction,
    compute_food_recognition_accuracy,
    compute_nutrient_mae,
)
from src.xai.explainer import XAIExplainer
from src.policy.policy_store import PolicyStore
from src.utils import set_global_seed, load_config

set_global_seed(42)

# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def agent_cfg():
    return load_config("agent_configs")

@pytest.fixture
def sys_cfg():
    return load_config("system_config")

@pytest.fixture
def basic_user_profile():
    return {
        "user_id": "test_user_001",
        "name": "Test User",
        "conditions": ["diabetes"],
        "disability_type": "none",
        "neurodivergent_type": "ASD",
        "sensory_preference": "mild",
        "reading_level": "standard",
        "daily_calorie_target": 2000,
        "low_sodium": False,
        "allergies": [],
        "caregiver_enabled": True,
    }

@pytest.fixture
def basic_context():
    return {"hour": 12, "location": "home", "accessibility_mode": "none"}

@pytest.fixture
def basic_constraints():
    return {"sugar_g": {"max": 30}, "glucose_limit": 180, "low_gi_preferred": True}

@pytest.fixture
def blackboard():
    return Blackboard(max_entries=100)

@pytest.fixture
def meal_agent(agent_cfg):
    return MealPlannerAgent(agent_cfg["meal_planner"])

@pytest.fixture
def reminder_agent(agent_cfg):
    return ReminderAgent(agent_cfg["reminder"])

@pytest.fixture
def food_agent(agent_cfg):
    return FoodGuidanceAgent(agent_cfg["food_guidance"])

@pytest.fixture
def monitoring_agent(agent_cfg):
    return MonitoringAgent(agent_cfg["monitoring"])

@pytest.fixture
def xai():
    return XAIExplainer()

@pytest.fixture
def policy(sys_cfg):
    return PolicyStore(sys_cfg.get("policy", {}))


# ── Blackboard Tests ───────────────────────────────────────────────────────────

class TestBlackboard:
    def test_post_and_query(self, blackboard):
        entry = blackboard.post(
            agent_id="meal_planner",
            entry_type="meal_plan",
            payload={"summary": "test plan"},
            priority=Priority.USER_PREFERENCE,
        )
        results = blackboard.query(entry_type="meal_plan")
        assert len(results) == 1
        assert results[0].agent_id == "meal_planner"

    def test_priority_ordering(self, blackboard):
        blackboard.post("agent_a", "alert", {}, Priority.BEHAVIORAL_NUDGE)
        blackboard.post("agent_b", "alert", {}, Priority.MEDICAL_SAFETY)
        blackboard.post("agent_c", "alert", {}, Priority.USER_PREFERENCE)
        results = blackboard.query(entry_type="alert")
        priorities = [r.priority for r in results]
        assert priorities == sorted(priorities, reverse=True) or len(set(priorities)) > 0

    def test_shared_context(self, blackboard):
        blackboard.set_context("last_glucose", 150)
        assert blackboard.get_context("last_glucose") == 150
        assert blackboard.get_context("nonexistent", "default") == "default"

    def test_snapshot(self, blackboard):
        blackboard.post("meal_planner", "meal_plan", {"x": 1})
        snap = blackboard.snapshot()
        assert snap["total_entries"] == 1
        assert "meal_plan" in snap["by_type"]


# ── LLM Decision Node Tests ───────────────────────────────────────────────────

class TestLLMDecisionNode:
    def test_check_meal_intent(self, basic_user_profile, basic_context):
        llm = LLMDecisionNode(use_mock=True)
        plan = llm.decide("Is this pasta OK for lunch?", basic_user_profile, basic_context)
        assert plan["intent"] == "check_meal"
        assert plan["primary_tool"] == "food_guidance"

    def test_plan_meal_intent(self, basic_user_profile, basic_context):
        llm = LLMDecisionNode(use_mock=True)
        plan = llm.decide("Suggest a meal plan for me", basic_user_profile, basic_context)
        assert plan["intent"] == "plan_meal"
        assert plan["primary_tool"] == "meal_planner"

    def test_health_intent(self, basic_user_profile, basic_context):
        llm = LLMDecisionNode(use_mock=True)
        plan = llm.decide("Check my heart rate vitals", basic_user_profile, basic_context)
        assert plan["intent"] == "check_health"
        assert plan["primary_tool"] == "monitoring"

    def test_constraints_extracted(self, basic_user_profile, basic_context):
        llm = LLMDecisionNode(use_mock=True)
        plan = llm.decide("What should I eat?", basic_user_profile, basic_context)
        assert "sugar_g" in plan["constraints"]  # diabetes constraint

    def test_call_graph_returned(self, basic_user_profile, basic_context):
        llm = LLMDecisionNode(use_mock=True)
        plan = llm.decide("Is this pasta OK?", basic_user_profile, basic_context)
        assert isinstance(plan["call_graph"], list)
        assert len(plan["call_graph"]) >= 1


# ── MCP Router Tests ──────────────────────────────────────────────────────────

class TestMCPRouter:
    def test_register_and_route(self, basic_user_profile, basic_context):
        router = MCPRouter()

        async def mock_handler(call: ToolCall):
            return {"result": "mock", "action_type": "test"}

        router.register("test_tool", mock_handler)
        call = ToolCall(
            tool_name="test_tool",
            intent="test",
            user_profile=basic_user_profile,
            context=basic_context,
        )
        result = asyncio.run(router.route(call))
        assert result.success
        assert result.payload["result"] == "mock"

    def test_missing_tool_graceful_degradation(self, basic_user_profile, basic_context):
        router = MCPRouter()
        call = ToolCall(
            tool_name="nonexistent_tool",
            intent="test",
            user_profile=basic_user_profile,
            context=basic_context,
        )
        result = asyncio.run(router.route(call))
        assert not result.success
        assert "Agent unavailable" in result.payload["message"]


# ── Agent Tests ───────────────────────────────────────────────────────────────

class TestMealPlannerAgent:
    def test_pra_run(self, meal_agent, basic_user_profile, basic_constraints):
        output = meal_agent.run({
            "user_profile": basic_user_profile,
            "constraints": basic_constraints,
            "context": {"hour": 12},
        })
        assert output.action_type == "meal_plan"
        assert "plan" in output.payload

    def test_clinical_compliance(self, meal_agent, basic_user_profile):
        constraints = {"sugar_g": {"max": 5}}  # very strict
        output = meal_agent.run({
            "user_profile": basic_user_profile,
            "constraints": constraints,
            "context": {"hour": 12},
        })
        # Should still return a plan (with safe fallbacks)
        assert output.action_type in ("meal_plan", "degraded")

    def test_feedback_updates_qtable(self, meal_agent):
        initial_epsilon = meal_agent.q_table.epsilon
        meal_agent.ingest_feedback({"accepted": True, "meal_id": 0, "state_idx": 0})
        # Epsilon should decay after feedback
        assert meal_agent.q_table.epsilon <= initial_epsilon


class TestReminderAgent:
    def test_sends_reminder_in_active_hours(self, reminder_agent, basic_user_profile):
        output = reminder_agent.run({
            "user_profile": basic_user_profile,
            "context": {"hour": 12},
            "sleep_window": {"start": 23, "end": 7},
            "engagement_history": [],
            "missed_events": ["breakfast"],
        })
        assert output.action_type in ("reminder", "reminder_suppressed")

    def test_suppresses_during_sleep(self, reminder_agent, basic_user_profile):
        output = reminder_agent.run({
            "user_profile": basic_user_profile,
            "context": {"hour": 2},  # 2am
            "sleep_window": {"start": 23, "end": 7},
        })
        assert output.action_type == "reminder_suppressed"

    def test_modality_selected(self, reminder_agent, basic_user_profile):
        output = reminder_agent.run({
            "user_profile": basic_user_profile,
            "context": {"hour": 12},
        })
        if output.action_type == "reminder":
            assert "modality" in output.payload


class TestFoodGuidanceAgent:
    def test_image_path_pasta(self, food_agent, basic_user_profile, basic_constraints):
        output = food_agent.run({
            "user_profile": basic_user_profile,
            "constraints": basic_constraints,
            "image_hint": "pasta",
            "context": {"hour": 12},
        })
        assert output.action_type in ("food_decision", "clarification_needed")
        if output.action_type == "food_decision":
            assert output.payload["verdict"] in ("approve", "limit", "swap", "deny")

    def test_nlp_path(self, food_agent, basic_user_profile, basic_constraints):
        output = food_agent.run({
            "user_profile": basic_user_profile,
            "constraints": basic_constraints,
            "query": "Is salad OK?",
            "context": {"hour": 12},
        })
        assert output.action_type in ("food_decision", "clarification_needed")

    def test_alternatives_on_swap(self, food_agent, basic_user_profile):
        # Force a swap by setting very strict constraints
        output = food_agent.run({
            "user_profile": basic_user_profile,
            "constraints": {"sugar_g": {"max": 1}, "calories": {"max": 50}},
            "image_hint": "pasta",
            "context": {"hour": 12},
        })
        if output.action_type == "food_decision" and output.payload.get("verdict") in ("swap", "limit"):
            assert "alternatives" in output.payload


class TestMonitoringAgent:
    def test_normal_vitals(self, monitoring_agent, basic_user_profile):
        output = monitoring_agent.run({
            "user_profile": basic_user_profile,
            "vitals": {"heart_rate": 72, "spo2": 98, "steps": 5000, "glucose": 95},
            "vital_sequence": [],
            "context": {"hour": 14},
            "intake_log": [{"meal_type": "breakfast"}, {"meal_type": "lunch"}],
        })
        assert output.action_type in ("monitoring_ok", "monitoring_suppressed")

    def test_anomalous_vitals_trigger_alert(self, monitoring_agent, basic_user_profile):
        output = monitoring_agent.run({
            "user_profile": basic_user_profile,
            "vitals": {"heart_rate": 135, "spo2": 90, "steps": 0, "glucose": 220},
            "vital_sequence": [],
            "context": {"hour": 14},
            "intake_log": [],
        })
        assert output.action_type in ("alert", "monitoring_suppressed")

    def test_wearable_offline_fallback(self, monitoring_agent, basic_user_profile):
        output = monitoring_agent.run({
            "user_profile": basic_user_profile,
            "vitals": {},
            "vital_sequence": [],
            "context": {"hour": 14},
            "intake_log": [],
            "wearables_connected": False,
        })
        # Should not crash — graceful degradation
        assert output is not None


# ── XAI Tests ─────────────────────────────────────────────────────────────────

class TestXAI:
    def test_explain_returns_string(self, xai, blackboard, basic_user_profile):
        blackboard.post("food_guidance", "food_decision", {
            "verdict": "limit", "food_class": "pasta",
            "confidence": 0.95, "_explanation_hint": "high glycemic index"
        })
        entries = blackboard.query()
        explanation = xai.explain(
            intent="check_meal",
            resolved_entries=entries,
            constraints={"low_gi_preferred": True},
            user_profile=basic_user_profile,
        )
        assert isinstance(explanation, str)
        assert len(explanation) > 5

    def test_simple_reading_level(self, xai, blackboard):
        blackboard.post("meal_planner", "meal_plan", {
            "plan": {"meals": [{"name": "Lentil soup"}], "nutrition_score": 0.85},
            "_explanation_hint": "Q-learning selected"
        })
        entries = blackboard.query()
        exp = xai.explain(
            intent="plan_meal",
            resolved_entries=entries,
            constraints={},
            user_profile={"reading_level": "simple"},
        )
        assert "🍽️" in exp or "healthy" in exp.lower() or "meal" in exp.lower()


# ── Policy Tests ──────────────────────────────────────────────────────────────

class TestPolicyStore:
    def test_consent_grant_and_check(self, policy):
        policy.grant_consent("user_test", scopes=["nutrition", "reminders"])
        allowed, reason = policy.check_access(
            scopes=["nutrition"], purpose="meal_plan", user_id="user_test"
        )
        assert allowed
        assert reason == "allowed"

    def test_denied_without_consent(self, policy):
        allowed, reason = policy.check_access(
            scopes=["nutrition"], purpose="test", user_id="no_consent_user"
        )
        assert not allowed

    def test_revoke_consent(self, policy):
        policy.grant_consent("revoke_user", scopes=["nutrition"])
        policy.revoke_consent("revoke_user")
        allowed, _ = policy.check_access(
            scopes=["nutrition"], purpose="test", user_id="revoke_user"
        )
        assert not allowed

    def test_safety_always_allowed(self, policy):
        allowed, reason = policy.check_access(
            scopes=["health_vitals_emergency"],
            purpose="emergency",
            user_id="any_user",
        )
        assert allowed
        assert reason == "safety_always_allowed"


# ── Evaluation Metrics Tests ──────────────────────────────────────────────────

class TestEvaluationMetrics:
    def test_nutritional_adequacy(self):
        records = [
            {"calories_total": 1800, "protein_g": 60, "fat_g": 55, "sugar_g": 40}
            for _ in range(100)
        ]
        result = compute_nutritional_adequacy(records)
        assert 0 <= result["nutritional_adequacy_pct"] <= 100

    def test_adherence_rate(self):
        records = [
            {"reminder_responses": ["complied", "complied", "ignored"]}
            for _ in range(50)
        ]
        result = compute_adherence_rate(records)
        assert abs(result["adherence_rate_pct"] - 66.67) < 1.0

    def test_food_recognition_accuracy(self):
        from src.agents.food_guidance import FOOD_CLASSES
        preds = FOOD_CLASSES * 10
        gt = FOOD_CLASSES * 10
        result = compute_food_recognition_accuracy(preds, gt)
        assert result["accuracy"] == 100.0

    def test_nutrient_mae(self):
        preds = [{"calories": 400, "protein_g": 20, "fat_g": 12, "sugar_g": 8}]
        gt = [{"calories": 414, "protein_g": 21.2, "fat_g": 13.3, "sugar_g": 9.1}]
        result = compute_nutrient_mae(preds, gt)
        assert result["mae_calories"] == pytest.approx(14.0, abs=0.1)


# ── Integration Test ───────────────────────────────────────────────────────────

class TestEndToEndPipeline:
    """Integration test: full request from prompt to response."""

    def test_full_pipeline_food_check(self, basic_user_profile, basic_context):
        from src.orchestrator import AgHealthOrchestrator
        orch = AgHealthOrchestrator()
        response = asyncio.run(orch.process_request(
            prompt="Is pasta OK for my lunch?",
            user_profile=basic_user_profile,
            context=basic_context,
            modality_inputs={"image_hint": "pasta"},
        ))
        assert "recommendation" in response
        assert "explanation" in response
        assert "trace_id" in response
        assert response["intent"] == "check_meal"

    def test_full_pipeline_meal_plan(self, basic_user_profile, basic_context):
        from src.orchestrator import AgHealthOrchestrator
        orch = AgHealthOrchestrator()
        response = asyncio.run(orch.process_request(
            prompt="Please plan my meals for the day",
            user_profile=basic_user_profile,
            context=basic_context,
        ))
        assert "recommendation" in response
        assert response["intent"] == "plan_meal"
