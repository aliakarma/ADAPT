"""
AgHealth+ — FastAPI REST API
==============================
Exposes the agentic system via HTTP endpoints for mobile/web integration.

Endpoints
---------
POST /api/v1/query          — Main request endpoint (full PRA pipeline)
POST /api/v1/feedback       — Feedback submission
GET  /api/v1/dashboard      — Blackboard + audit snapshot
GET  /api/v1/health         — System health check
POST /api/v1/consent        — Register user consent
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from loguru import logger

from src.orchestrator import AgHealthOrchestrator
from src.utils import load_config


# ──────────────────────────────────────────────────────────────────────────────
# App
# ──────────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="AgHealth+ API",
    description="Agentic AI Nutrition & Healthcare Monitor for Neurodivergent and Disabled Users",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global orchestrator (initialised at startup)
_orchestrator: Optional[AgHealthOrchestrator] = None


@app.on_event("startup")
async def startup():
    global _orchestrator
    logger.info("API | startup | initialising AgHealth+ orchestrator...")
    _orchestrator = AgHealthOrchestrator()
    logger.info("API | ready")


def get_orchestrator() -> AgHealthOrchestrator:
    if _orchestrator is None:
        raise HTTPException(status_code=503, detail="Orchestrator not initialised")
    return _orchestrator


# ──────────────────────────────────────────────────────────────────────────────
# Request / Response Models
# ──────────────────────────────────────────────────────────────────────────────

class UserProfileModel(BaseModel):
    user_id: str
    name: str = "User"
    age: int = 30
    conditions: List[str] = Field(default_factory=list)
    neurodivergent_type: str = "none"
    disability_type: str = "none"
    sensory_preference: str = "mild"
    reading_level: str = "standard"
    daily_calorie_target: int = 2000
    low_sodium: bool = False
    calorie_restriction: bool = False
    allergies: List[str] = Field(default_factory=list)
    caregiver_enabled: bool = False


class QueryRequest(BaseModel):
    prompt: str = Field(..., description="User query (text or voice transcription)")
    user_profile: UserProfileModel
    context: Dict[str, Any] = Field(default_factory=dict)
    modality_inputs: Dict[str, Any] = Field(
        default_factory=dict,
        description="Extra inputs: image_hint, vitals, intake_log, etc."
    )
    trace_id: Optional[str] = None


class FeedbackRequest(BaseModel):
    agent_id: str
    trace_id: str
    feedback: Dict[str, Any]


class ConsentRequest(BaseModel):
    user_id: str
    scopes: List[str]
    caregiver_updates: bool = False
    retention_days: int = 90


# ──────────────────────────────────────────────────────────────────────────────
# Endpoints
# ──────────────────────────────────────────────────────────────────────────────

@app.get("/api/v1/health")
async def health_check():
    return {"status": "ok", "service": "AgHealth+", "version": "1.0.0"}


@app.post("/api/v1/consent")
async def register_consent(req: ConsentRequest):
    orch = get_orchestrator()
    record = orch.policy.grant_consent(
        user_id=req.user_id,
        scopes=req.scopes,
        caregiver_updates=req.caregiver_updates,
        retention_days=req.retention_days,
    )
    return {"status": "consent_granted", "record": record.to_dict()}


@app.post("/api/v1/query")
async def process_query(req: QueryRequest):
    """
    Main query endpoint — runs the full 6-step PRA pipeline.

    Example body:
    {
      "prompt": "Is this pasta OK for lunch?",
      "user_profile": {"user_id": "u001", "conditions": ["diabetes"]},
      "context": {"hour": 12},
      "modality_inputs": {"image_hint": "pasta"}
    }
    """
    orch = get_orchestrator()
    try:
        response = await orch.process_request(
            prompt=req.prompt,
            user_profile=req.user_profile.model_dump(),
            context=req.context,
            modality_inputs=req.modality_inputs,
            trace_id=req.trace_id,
        )
        return response
    except Exception as exc:
        logger.exception("API | /query error: {}", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/api/v1/feedback")
async def submit_feedback(req: FeedbackRequest):
    orch = get_orchestrator()
    orch.send_feedback(req.agent_id, {**req.feedback, "trace_id": req.trace_id})
    return {"status": "feedback_received"}


@app.get("/api/v1/dashboard")
async def dashboard():
    orch = get_orchestrator()
    return orch.get_dashboard_snapshot()


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error("API | unhandled exception: {}", exc)
    return JSONResponse(status_code=500, content={"detail": str(exc)})
