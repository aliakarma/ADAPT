"""
ADAPT — Base Agent (PRA Loop)
====================================
All four specialized agents inherit from this abstract base class.

Each agent implements the Perception–Reasoning–Action (PRA) loop:
  - perceive(inputs)  → structured percept dict
  - reason(percept)   → decision / plan dict
  - act(decision)     → output dict posted to blackboard

The base class provides:
  - Lifecycle management (start / stop)
  - Standard logging and tracing
  - Graceful degradation on failure
  - Feedback ingestion for adaptive learning
"""
from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from loguru import logger

from ..utils.helpers import generate_trace_id, now_utc


# ──────────────────────────────────────────────────────────────────────────────
# PRA Percept / Decision containers
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class Percept:
    raw_inputs: Dict[str, Any]
    user_id: str
    trace_id: str = field(default_factory=generate_trace_id)
    timestamp: str = field(default_factory=now_utc)


@dataclass
class AgentOutput:
    agent_id: str
    action_type: str
    payload: Dict[str, Any]
    trace_id: str
    confidence: float = 1.0
    explanation_hint: str = ""    # forwarded to XAI
    timestamp: str = field(default_factory=now_utc)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "action_type": self.action_type,
            "confidence": self.confidence,
            "explanation_hint": self.explanation_hint,
            "timestamp": self.timestamp,
            **self.payload,
        }


# ──────────────────────────────────────────────────────────────────────────────
# Abstract Base Agent
# ──────────────────────────────────────────────────────────────────────────────

class BaseAgent(abc.ABC):
    """
    Abstract base class implementing the PRA loop.

    Subclasses must implement:
      - perceive(inputs) -> Dict
      - reason(percept)  -> Dict
      - act(decision)    -> Dict
    """

    def __init__(self, agent_id: str, config: Dict[str, Any]):
        self.agent_id = agent_id
        self.config = config
        self._running = False
        self._feedback_buffer: List[Dict[str, Any]] = []
        logger.info("Agent | {} | initialised", agent_id)

    # ── PRA pipeline ─────────────────────────────────────────────────────────

    def run(self, inputs: Dict[str, Any], trace_id: str = "") -> AgentOutput:
        """
        Execute one full PRA cycle.

        Parameters
        ----------
        inputs : dict
            Raw inputs from MCP ToolCall (user query, image bytes, sensor data, etc.)
        trace_id : str

        Returns
        -------
        AgentOutput
        """
        trace_id = trace_id or generate_trace_id()
        logger.debug("Agent | {} | PRA start | trace={}", self.agent_id, trace_id[:8])

        try:
            percept = self.perceive(inputs)
            decision = self.reason(percept)
            output_payload = self.act(decision)
        except Exception as exc:
            logger.error("Agent | {} | PRA error: {} | trace={}", self.agent_id, exc, trace_id[:8])
            output_payload = self._graceful_degradation(exc)
            return AgentOutput(
                agent_id=self.agent_id,
                action_type="degraded",
                payload=output_payload,
                trace_id=trace_id,
                confidence=0.0,
                explanation_hint=f"Agent degraded due to: {str(exc)[:100]}",
            )

        confidence = output_payload.pop("_confidence", 1.0)
        explanation_hint = output_payload.pop("_explanation_hint", "")

        result = AgentOutput(
            agent_id=self.agent_id,
            action_type=output_payload.pop("action_type", "unknown"),
            payload=output_payload,
            trace_id=trace_id,
            confidence=confidence,
            explanation_hint=explanation_hint,
        )
        logger.debug("Agent | {} | PRA done | action={} trace={}", self.agent_id, result.action_type, trace_id[:8])
        return result

    # ── Abstract PRA methods ──────────────────────────────────────────────────

    @abc.abstractmethod
    def perceive(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Transform raw inputs into a structured percept."""

    @abc.abstractmethod
    def reason(self, percept: Dict[str, Any]) -> Dict[str, Any]:
        """Deliberate on the percept and produce a decision/plan."""

    @abc.abstractmethod
    def act(self, decision: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the decision and return an output dict."""

    # ── Feedback / learning ──────────────────────────────────────────────────

    def ingest_feedback(self, feedback: Dict[str, Any]) -> None:
        """
        Receive a reinforcement signal (e.g., user accepted/rejected a plan).
        Stored for batch learning; subclasses may override to update online.
        """
        self._feedback_buffer.append({"timestamp": now_utc(), **feedback})
        logger.debug("Agent | {} | feedback ingested (buffer={})", self.agent_id, len(self._feedback_buffer))

    def learn_from_feedback(self) -> None:
        """
        Default: no-op.  Subclasses with online learning override this.
        Called periodically by the orchestrator.
        """

    # ── Lifecycle ────────────────────────────────────────────────────────────

    def start(self) -> None:
        self._running = True
        logger.info("Agent | {} | started", self.agent_id)

    def stop(self) -> None:
        self._running = False
        logger.info("Agent | {} | stopped", self.agent_id)

    # ── Graceful degradation ─────────────────────────────────────────────────

    def _graceful_degradation(self, exc: Exception) -> Dict[str, Any]:
        return {
            "message": "Agent temporarily unavailable. Core routines continue.",
            "error": str(exc)[:200],
        }
