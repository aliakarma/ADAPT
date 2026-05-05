"""
AgHealth+ — Policy Store
==========================
Implements least-privilege data access, consent management, and audit logging.

All agent data requests pass through this store before retrieval.
Every write operation is logged with: purpose, scope, retention, and trace_id.
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from loguru import logger

from ..utils.helpers import now_utc, generate_trace_id


# ──────────────────────────────────────────────────────────────────────────────
# Consent Record
# ──────────────────────────────────────────────────────────────────────────────

class ConsentRecord:
    def __init__(
        self,
        user_id: str,
        granted_scopes: List[str],
        caregiver_updates: bool = False,
        retention_days: int = 90,
    ):
        self.user_id = user_id
        self.granted_scopes = set(granted_scopes)
        self.caregiver_updates = caregiver_updates
        self.retention_days = retention_days
        self.granted_at = now_utc()
        self.revoked = False
        self.revoked_at: Optional[str] = None

    def revoke(self) -> None:
        self.revoked = True
        self.revoked_at = now_utc()
        logger.info("Consent | revoked | user={}", self.user_id[:8])

    def has_scope(self, scope: str) -> bool:
        return not self.revoked and scope in self.granted_scopes

    def to_dict(self) -> Dict[str, Any]:
        return {
            "user_id": self.user_id,
            "granted_scopes": list(self.granted_scopes),
            "caregiver_updates": self.caregiver_updates,
            "retention_days": self.retention_days,
            "granted_at": self.granted_at,
            "revoked": self.revoked,
            "revoked_at": self.revoked_at,
        }


# ──────────────────────────────────────────────────────────────────────────────
# Audit Entry
# ──────────────────────────────────────────────────────────────────────────────

class AuditEntry:
    def __init__(
        self,
        trace_id: str,
        user_id: str,
        agent_id: str,
        data_type: str,
        purpose: str,
        scope: str,
        allowed: bool,
        retention_days: int,
    ):
        self.trace_id = trace_id
        self.user_id = user_id
        self.agent_id = agent_id
        self.data_type = data_type
        self.purpose = purpose
        self.scope = scope
        self.allowed = allowed
        self.retention_days = retention_days
        self.timestamp = now_utc()

    def to_dict(self) -> Dict[str, Any]:
        return self.__dict__


# ──────────────────────────────────────────────────────────────────────────────
# Policy Store
# ──────────────────────────────────────────────────────────────────────────────

class PolicyStore:
    """
    Enforces:
    - Least-privilege data access (scope checks)
    - Consent management (per-user opt-in/out)
    - Audit trail (every access logged with purpose + retention)
    - Medical safety priority (always allowed regardless of consent)
    """

    # Scopes that are always allowed (safety-critical)
    SAFETY_ALWAYS_ALLOWED = {"health_vitals_emergency", "medication_alert"}

    def __init__(self, config: Dict[str, Any], audit_log_path: str = "results/logs/audit.jsonl"):
        self.config = config
        self.audit_log_path = audit_log_path
        self._consents: Dict[str, ConsentRecord] = {}
        self._audit: List[AuditEntry] = []
        Path(audit_log_path).parent.mkdir(parents=True, exist_ok=True)

    # ── Consent management ────────────────────────────────────────────────────

    def grant_consent(
        self,
        user_id: str,
        scopes: List[str],
        caregiver_updates: bool = False,
        retention_days: int = 90,
    ) -> ConsentRecord:
        record = ConsentRecord(user_id, scopes, caregiver_updates, retention_days)
        self._consents[user_id] = record
        logger.info("PolicyStore | consent granted | user={} scopes={}", user_id[:8], scopes)
        return record

    def revoke_consent(self, user_id: str) -> None:
        if user_id in self._consents:
            self._consents[user_id].revoke()

    def get_consent(self, user_id: str) -> Optional[ConsentRecord]:
        return self._consents.get(user_id)

    # ── Access control ────────────────────────────────────────────────────────

    def check_access(
        self,
        scopes: List[str],
        purpose: str,
        user_id: str,
        agent_id: str = "system",
        trace_id: str = "",
    ) -> Tuple[bool, str]:
        """
        Check whether the requested scopes are permitted for this user.

        Returns (allowed: bool, reason: str)
        """
        trace_id = trace_id or generate_trace_id()

        # Safety-critical scopes are always allowed
        if any(s in self.SAFETY_ALWAYS_ALLOWED for s in scopes):
            self._log_access(trace_id, user_id, agent_id, scopes, purpose, True, 30)
            return True, "safety_always_allowed"

        consent = self._consents.get(user_id)

        # If consent-required mode and no consent record → deny
        if self.config.get("require_explicit_consent", True) and consent is None:
            self._log_access(trace_id, user_id, agent_id, scopes, purpose, False, 0)
            return False, "no_consent_record"

        # If consent is revoked → deny
        if consent and consent.revoked:
            self._log_access(trace_id, user_id, agent_id, scopes, purpose, False, 0)
            return False, "consent_revoked"

        # Check each scope
        if consent:
            denied_scopes = [s for s in scopes if not consent.has_scope(s)]
            if denied_scopes:
                self._log_access(trace_id, user_id, agent_id, scopes, purpose, False,
                                  consent.retention_days)
                return False, f"scope_denied: {denied_scopes}"
            retention = consent.retention_days
        else:
            retention = self.config.get("data_retention_days", 90)

        self._log_access(trace_id, user_id, agent_id, scopes, purpose, True, retention)
        return True, "allowed"

    # ── Data retrieval (controlled) ───────────────────────────────────────────

    def retrieve_data(
        self,
        data_type: str,
        user_id: str,
        data_store: Dict[str, Any],
        purpose: str = "",
        trace_id: str = "",
    ) -> Optional[Any]:
        """
        Retrieve data only after policy check.
        data_store is the application's data layer dict.
        """
        allowed, reason = self.check_access(
            scopes=[data_type],
            purpose=purpose,
            user_id=user_id,
            trace_id=trace_id,
        )
        if not allowed:
            logger.warning("PolicyStore | data access denied | type={} user={} reason={}",
                           data_type, user_id[:8], reason)
            return None
        return data_store.get(data_type)

    # ── Audit ─────────────────────────────────────────────────────────────────

    def _log_access(
        self,
        trace_id: str,
        user_id: str,
        agent_id: str,
        scopes: List[str],
        purpose: str,
        allowed: bool,
        retention_days: int,
    ) -> None:
        entry = AuditEntry(
            trace_id=trace_id,
            user_id=user_id[:12],   # truncate for privacy
            agent_id=agent_id,
            data_type=",".join(scopes),
            purpose=purpose[:100],
            scope=",".join(scopes),
            allowed=allowed,
            retention_days=retention_days,
        )
        self._audit.append(entry)

        # Append to JSONL file for external audit
        with open(self.audit_log_path, "a") as f:
            f.write(json.dumps(entry.to_dict()) + "\n")

    def audit_log(self, user_id: Optional[str] = None) -> List[Dict[str, Any]]:
        entries = self._audit
        if user_id:
            entries = [e for e in entries if e.user_id == user_id[:12]]
        return [e.to_dict() for e in entries]

    def audit_summary(self) -> Dict[str, Any]:
        total = len(self._audit)
        allowed = sum(1 for e in self._audit if e.allowed)
        return {
            "total_checks": total,
            "allowed": allowed,
            "denied": total - allowed,
            "allow_rate": allowed / total if total > 0 else 0.0,
        }
