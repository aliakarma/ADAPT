"""
AgHealth+ — Central Blackboard
================================
Implements a thread-safe, priority-aware shared knowledge base that all agents
read/write to. The blackboard follows the classic BB architectural pattern:

  - Knowledge sources (agents) post entries (observations, plans, alerts).
  - The Reasoning Core reads and merges entries, resolving conflicts by
    priority: medical_safety > user_preference > behavioral_nudge.
  - Each entry carries: agent_id, type, payload, priority, trace_id, timestamp.
"""
import threading
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import IntEnum
from typing import Any, Dict, List, Optional

from loguru import logger


# ──────────────────────────────────────────────────────────────────────────────
# Priority levels (higher = more important)
# ──────────────────────────────────────────────────────────────────────────────

class Priority(IntEnum):
    BEHAVIORAL_NUDGE = 1
    USER_PREFERENCE = 2
    MEDICAL_SAFETY = 3


# ──────────────────────────────────────────────────────────────────────────────
# Blackboard Entry
# ──────────────────────────────────────────────────────────────────────────────

@dataclass(order=True)
class BlackboardEntry:
    priority: int
    timestamp: str = field(compare=False)
    agent_id: str = field(compare=False)
    entry_type: str = field(compare=False)   # e.g. "meal_plan", "alert", "reminder"
    payload: Dict[str, Any] = field(compare=False, default_factory=dict)
    trace_id: str = field(compare=False, default="")
    resolved: bool = field(compare=False, default=False)


# ──────────────────────────────────────────────────────────────────────────────
# Blackboard
# ──────────────────────────────────────────────────────────────────────────────

class Blackboard:
    """
    Central shared knowledge base.

    Thread-safe via a reentrant lock. Entries expire after `ttl_seconds`
    to prevent stale context from influencing decisions.
    """

    def __init__(self, max_entries: int = 500, ttl_seconds: int = 3600):
        self._lock = threading.RLock()
        self._entries: List[BlackboardEntry] = []
        self._max_entries = max_entries
        self._ttl_seconds = ttl_seconds
        self._shared_context: Dict[str, Any] = {}

    # ── Write ────────────────────────────────────────────────────────────────

    def post(
        self,
        agent_id: str,
        entry_type: str,
        payload: Dict[str, Any],
        priority: Priority = Priority.USER_PREFERENCE,
        trace_id: str = "",
    ) -> BlackboardEntry:
        entry = BlackboardEntry(
            priority=int(priority),
            timestamp=datetime.now(timezone.utc).isoformat(),
            agent_id=agent_id,
            entry_type=entry_type,
            payload=payload,
            trace_id=trace_id,
        )
        with self._lock:
            self._entries.append(entry)
            self._evict()
        logger.debug(
            "Blackboard | post | agent={} type={} priority={} trace={}",
            agent_id, entry_type, priority.name, trace_id[:8]
        )
        return entry

    def set_context(self, key: str, value: Any) -> None:
        """Store a key-value pair in the shared context dict."""
        with self._lock:
            self._shared_context[key] = value

    def get_context(self, key: str, default: Any = None) -> Any:
        with self._lock:
            return self._shared_context.get(key, default)

    # ── Read ─────────────────────────────────────────────────────────────────

    def query(
        self,
        entry_type: Optional[str] = None,
        agent_id: Optional[str] = None,
        min_priority: Priority = Priority.BEHAVIORAL_NUDGE,
        resolved: Optional[bool] = None,
        limit: int = 50,
    ) -> List[BlackboardEntry]:
        """Return entries matching filters, sorted by priority desc."""
        with self._lock:
            results = [
                e for e in self._entries
                if (entry_type is None or e.entry_type == entry_type)
                and (agent_id is None or e.agent_id == agent_id)
                and e.priority >= int(min_priority)
                and (resolved is None or e.resolved == resolved)
            ]
        results.sort(key=lambda e: (-e.priority, e.timestamp), reverse=False)
        return results[:limit]

    def latest(self, entry_type: str) -> Optional[BlackboardEntry]:
        """Return the most recent entry of a given type."""
        matches = self.query(entry_type=entry_type, limit=1)
        return matches[0] if matches else None

    def mark_resolved(self, entry: BlackboardEntry) -> None:
        with self._lock:
            entry.resolved = True

    # ── Snapshot ─────────────────────────────────────────────────────────────

    def snapshot(self) -> Dict[str, Any]:
        """Serialisable summary for XAI and dashboard."""
        with self._lock:
            return {
                "total_entries": len(self._entries),
                "unresolved": sum(1 for e in self._entries if not e.resolved),
                "by_type": self._count_by("entry_type"),
                "by_agent": self._count_by("agent_id"),
                "shared_context_keys": list(self._shared_context.keys()),
            }

    def _count_by(self, attr: str) -> Dict[str, int]:
        counts: Dict[str, int] = defaultdict(int)
        for e in self._entries:
            counts[getattr(e, attr)] += 1
        return dict(counts)

    # ── Housekeeping ─────────────────────────────────────────────────────────

    def _evict(self) -> None:
        """Remove oldest resolved entries when over capacity."""
        if len(self._entries) > self._max_entries:
            resolved = [e for e in self._entries if e.resolved]
            resolved.sort(key=lambda e: e.timestamp)
            for e in resolved[: len(self._entries) - self._max_entries]:
                self._entries.remove(e)

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()
            self._shared_context.clear()
