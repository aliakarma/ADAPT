"""
AgHealth+ — MCP Router (Model Context Protocol)
================================================
The MCP Router acts as a **context bridge** between the LLM Decision Node and
the downstream specialized agents (or external connectors).

Responsibilities
----------------
1. Receive an *intent plan* from the LLM Decision Node.
2. Package the intent with user profile, contextual metadata, and policy scope
   into a typed ToolCall.
3. Route the ToolCall to the correct agent or external connector.
4. Return a ToolResult with trace metadata for XAI.
5. Enforce purpose + scope + retention checks on every call (per policy store).
"""
import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional

from loguru import logger

from ..utils.helpers import generate_trace_id, now_utc


# ──────────────────────────────────────────────────────────────────────────────
# Data classes
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class ToolCall:
    tool_name: str                  # e.g. "food_guidance", "meal_planner"
    intent: str                     # parsed intent string
    user_profile: Dict[str, Any]
    context: Dict[str, Any]         # time, location, accessibility mode, etc.
    payload: Dict[str, Any] = field(default_factory=dict)  # image bytes, sensor ref, etc.
    scope: List[str] = field(default_factory=list)          # data scopes requested
    trace_id: str = field(default_factory=generate_trace_id)
    timestamp: str = field(default_factory=now_utc)


@dataclass
class ToolResult:
    tool_name: str
    success: bool
    payload: Dict[str, Any]
    trace_id: str
    latency_ms: float
    error: Optional[str] = None
    external_refs: List[str] = field(default_factory=list)  # for XAI


# ──────────────────────────────────────────────────────────────────────────────
# MCP Router
# ──────────────────────────────────────────────────────────────────────────────

class MCPRouter:
    """
    Translates LLM intent plans into agent tool calls and returns results
    with full trace metadata.

    Usage
    -----
    router = MCPRouter(policy_store)
    router.register("food_guidance", food_agent.handle)
    result = await router.route(tool_call)
    """

    def __init__(self, policy_store=None, timeout: float = 10.0):
        self._registry: Dict[str, Callable] = {}
        self._policy_store = policy_store
        self._timeout = timeout
        self._call_log: List[Dict[str, Any]] = []

    # ── Registry ─────────────────────────────────────────────────────────────

    def register(self, tool_name: str, handler: Callable) -> None:
        """Register an agent handler under a tool name."""
        self._registry[tool_name] = handler
        logger.info("MCPRouter | registered tool: {}", tool_name)

    def registered_tools(self) -> List[str]:
        return list(self._registry.keys())

    # ── Routing ──────────────────────────────────────────────────────────────

    async def route(self, call: ToolCall) -> ToolResult:
        """
        Route a ToolCall to the appropriate handler.
        1. Policy check
        2. Dispatch (with timeout)
        3. Return ToolResult with trace metadata
        """
        import time as _time
        start = _time.perf_counter()

        logger.info(
            "MCPRouter | route | tool={} intent='{}' trace={}",
            call.tool_name, call.intent, call.trace_id[:8]
        )

        # 1. Policy check
        if self._policy_store:
            allowed, reason = self._policy_store.check_access(
                scopes=call.scope,
                purpose=call.intent,
                user_id=call.user_profile.get("user_id", ""),
            )
            if not allowed:
                return ToolResult(
                    tool_name=call.tool_name,
                    success=False,
                    payload={},
                    trace_id=call.trace_id,
                    latency_ms=0.0,
                    error=f"Policy denied: {reason}",
                )

        # 2. Dispatch
        handler = self._registry.get(call.tool_name)
        if handler is None:
            # Graceful degradation — return empty result, log the miss
            logger.warning("MCPRouter | tool '{}' not registered — degraded response", call.tool_name)
            return ToolResult(
                tool_name=call.tool_name,
                success=False,
                payload={"message": "Agent unavailable; core routines unaffected."},
                trace_id=call.trace_id,
                latency_ms=0.0,
                error="Tool not registered",
            )

        try:
            if asyncio.iscoroutinefunction(handler):
                result_payload = await asyncio.wait_for(handler(call), timeout=self._timeout)
            else:
                result_payload = handler(call)
        except asyncio.TimeoutError:
            logger.error("MCPRouter | timeout | tool={} trace={}", call.tool_name, call.trace_id[:8])
            return ToolResult(
                tool_name=call.tool_name,
                success=False,
                payload={},
                trace_id=call.trace_id,
                latency_ms=self._timeout * 1000,
                error="Handler timeout — graceful fallback applied",
            )
        except Exception as exc:
            logger.exception("MCPRouter | error | tool={} error={}", call.tool_name, exc)
            return ToolResult(
                tool_name=call.tool_name,
                success=False,
                payload={},
                trace_id=call.trace_id,
                latency_ms=((_time.perf_counter() - start) * 1000),
                error=str(exc),
            )

        latency = (_time.perf_counter() - start) * 1000
        result = ToolResult(
            tool_name=call.tool_name,
            success=True,
            payload=result_payload if isinstance(result_payload, dict) else {"data": result_payload},
            trace_id=call.trace_id,
            latency_ms=round(latency, 2),
        )

        # 3. Audit log
        self._call_log.append({
            "trace_id": call.trace_id,
            "tool": call.tool_name,
            "intent": call.intent,
            "success": result.success,
            "latency_ms": result.latency_ms,
            "timestamp": now_utc(),
        })

        logger.debug("MCPRouter | done | tool={} latency={:.1f}ms", call.tool_name, latency)
        return result

    def audit_log(self) -> List[Dict[str, Any]]:
        return list(self._call_log)
