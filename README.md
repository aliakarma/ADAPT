# ADAPT — Agentic AI Nutrition & Healthcare Monitor


**Production-ready implementation** of the paper:  
*"Agentic AI for Inclusive Nutrition and Healthcare: A Multi-Agent Framework for Neurodivergent and Disabled Users"*

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Architecture: Multi-Agent PRA](https://img.shields.io/badge/Architecture-Multi--Agent%20PRA-orange.svg)]()

---

## Overview

ADAPT is a layered, multi-agent AI system designed to help people with disabilities and neurodivergent conditions manage their nutrition and health independently. It implements the full **Perception–Reasoning–Action (PRA)** loop across four specialized agents, coordinated by a central blackboard and LLM decision node.

### Key Results (Simulated Pilot — 500 users × 8 weeks)

| Metric | System | Paper Reported |
|--------|--------|---------------|
| Food Recognition Accuracy | 99–100% | 99% |
| Nutrient MAE (Calories) | ~14 kcal | 13.9 kcal |
| Adherence Rate | 60–81% | 81% |
| User Satisfaction | 4.2–4.4/5 | 4.2/5 |
| Explainability Rate | 92–100% | 92% |
| Caregiver Burden Reduction | 35–84% | 35% |

---

## Architecture

```
User Input (voice / text / image)
        │
        ▼
┌─────────────────────┐
│  Multimodal UI      │  ← context normalisation, accessibility modes
└─────────┬───────────┘
          │ structured prompt
          ▼
┌─────────────────────┐
│  LLM Decision Node  │  ← intent parsing, call-graph generation
└─────────┬───────────┘
          │ intent plan
          ▼
┌─────────────────────┐
│  MCP Router         │  ← context bridge, policy checks, trace IDs
└──┬──────┬──────┬────┘
   │      │      │
   ▼      ▼      ▼
┌──────┐ ┌────┐ ┌──────────┐ ┌──────────┐
│Meal  │ │Rem-│ │Food      │ │Monitor-  │
│Planner│ │inder│ │Guidance  │ │ing Agent │
│(Q-RL)│ │(UCB1│ │(CNN+NLP) │ │(GRU)     │
└──┬───┘ └──┬─┘ └────┬─────┘ └────┬─────┘
   └────────┴─────────┴────────────┘
                     │ PRA outputs
                     ▼
        ┌────────────────────────┐
        │  Central Reasoning     │  ← conflict resolution, priority weighting
        │  Core + Blackboard     │  ← medical > preference > nudge
        └────────────┬───────────┘
                     │
                     ▼
           ┌─────────────────┐
           │  XAI Explainer  │  ← plain-language explanations
           └────────┬────────┘
                    │
        ┌───────────┴──────────┐
        │                      │
        ▼                      ▼
   User Card            Caregiver Dashboard
   (accessible)         (optional, permissioned)
```

### Four Specialized Agents (PRA Loops)

| Agent | Perception | Reasoning | Action |
|-------|-----------|-----------|--------|
| **Meal Planner** | Profile, EHR rules, recent intake | Q-learning over meal options | Daily/weekly menu + shopping list |
| **Reminder** | Engagement logs, sleep windows | UCB1 contextual bandit | Vibration / banner / icon / sound |
| **Food Guidance** | Image/barcode, NL query | CNN classification + plan comparison | Approve / Limit / Swap + cooking steps |
| **Monitoring** | Wearable vitals, kitchen sensors | GRU anomaly detection | Alert + caregiver notification |

---

## Installation

### Prerequisites
- Python 3.10+
- pip

### Setup

```bash
# 1. Clone repository
git clone https://github.com/aliakarma/ADAPT.git
cd aghealth-plus

# 2. Create virtual environment
python -m venv .venv
source .venv/bin/activate      # Linux/macOS
# .venv\Scripts\activate       # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. (Optional) Install as package
pip install -e .
```

### Docker

```bash
docker build -t aghealth-plus .
docker run -p 8000:8000 aghealth-plus
```

---

## Configuration

All configs are in `configs/`:

| File | Purpose |
|------|---------|
| `system_config.yaml` | System-level settings (logging, blackboard, MCP, LLM, policy) |
| `agent_configs.yaml` | Per-agent hyperparameters (Q-learning, bandit, CNN thresholds) |
| `model_configs.yaml` | ML model architectures and training settings |

**Offline mode** (default): `llm.use_mock: true` — fully rule-based, no API key needed.  
**API mode**: set `OPENAI_API_KEY` env variable and `llm.use_mock: false`.

---

## Quickstart

### Run the full pilot simulation

```bash
python experiments/run_pilot_simulation.py --n_users 500 --n_weeks 8 --seed 42
```

### Generate all paper figures

```bash
python experiments/visualise_results.py
# Outputs → results/graphs/
```

### Train models

```bash
# CNN food classifier
python experiments/train_cnn.py --epochs 20 --n_samples 5000

# GRU anomaly detector
python experiments/train_gru.py --epochs 30 --n_users 500
```

### Start REST API

```bash
uvicorn api.app:app --host 0.0.0.0 --port 8000
# Swagger UI: http://localhost:8000/docs
```

### Example API request

```bash
curl -X POST http://localhost:8000/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Is pasta OK for my lunch?",
    "user_profile": {
      "user_id": "u001",
      "conditions": ["diabetes"],
      "neurodivergent_type": "ASD",
      "daily_calorie_target": 2000
    },
    "context": {"hour": 12},
    "modality_inputs": {"image_hint": "pasta"}
  }'
```

### Run tests

```bash
pytest tests/ -v
```

---

## Project Structure

```
ADAPT/
├── src/
│   ├── agents/
│   │   ├── base_agent.py          # Abstract PRA base class
│   │   ├── meal_planner.py        # Q-learning meal recommendation
│   │   ├── reminder.py            # UCB1 contextual bandit reminders
│   │   ├── food_guidance.py       # CNN + NLP food recognition
│   │   └── monitoring.py          # GRU anomaly detection
│   ├── core/
│   │   ├── blackboard.py          # Thread-safe shared knowledge base
│   │   ├── mcp_router.py          # Context bridge & tool routing
│   │   ├── llm_decision_node.py   # Intent parsing & call-graph generation
│   │   └── reasoning_core.py      # Conflict resolution & coordination
│   ├── models/
│   │   ├── cnn_food_classifier.py # MobileNetV2 + nutrient regression
│   │   └── gru_anomaly.py         # GRU anomaly detection model
│   ├── data/
│   │   └── dataset_generator.py   # 500-user synthetic dataset
│   ├── evaluation/
│   │   └── metrics.py             # All paper evaluation metrics
│   ├── xai/
│   │   └── explainer.py           # Plain-language explanation module
│   ├── policy/
│   │   └── policy_store.py        # Consent, least-privilege, audit
│   └── orchestrator.py            # End-to-end pipeline coordinator
├── api/
│   └── app.py                     # FastAPI REST endpoints
├── configs/
│   ├── system_config.yaml
│   ├── agent_configs.yaml
│   └── model_configs.yaml
├── data/synthetic/                # Generated synthetic dataset
├── experiments/
│   ├── run_pilot_simulation.py    # Full pilot experiment
│   ├── visualise_results.py       # Paper figure reproduction
│   ├── train_cnn.py               # CNN training script
│   └── train_gru.py               # GRU training script
├── results/
│   ├── graphs/                    # All paper figures (PNG)
│   ├── tables/                    # Evaluation results (JSON)
│   └── logs/                      # System and audit logs
├── tests/
│   └── test_aghealth.py           # Unit + integration tests
├── requirements.txt
├── setup.py
└── Dockerfile
```

---

## Evaluation Metrics

| Metric | Definition | Paper Value |
|--------|-----------|-------------|
| Nutritional Adequacy | % daily plans meeting ≥80% DRI | +27% vs baseline |
| Adherence Rate | complied / total reminders | 54% → 81% |
| User Satisfaction | Likert 1–5 mean | 4.2/5 |
| Explainability Rate | % decisions with plain explanation | 92% |
| Caregiver Burden Reduction | Δ intervention rate | –35% |
| Food Recognition Accuracy | Top-1 CNN accuracy | 99% |
| Nutrient MAE (calories) | Mean absolute error | 13.9 kcal |

---

## Accessibility Features

The system is designed accessibility-first for:

- **Blind/low-vision**: Screen reader support, alt text, speech output
- **Deaf/hard-of-hearing**: Visual and tactile equivalents for all audio
- **Motor impairments**: Voice-first, large tap targets, hands-free flows
- **Cognitive disabilities**: Simple pictograms, step pauses, concrete language
- **ASD/ADHD**: Low-stimulation layouts, predictable structure, sensory-sensitive meal planning
- **Anxiety/depression**: Caring, nonjudgmental reminders; comfort food options within clinical limits

---

## Reproducibility

All experiments are fully reproducible with `--seed 42`:

```bash
python experiments/run_pilot_simulation.py --seed 42
python experiments/train_cnn.py --seed 42
python experiments/train_gru.py --seed 42
```

Results are saved to `results/tables/evaluation_results.json`.

---

## Safety & Governance

- Medical constraints always take highest priority (Priority.MEDICAL_SAFETY)
- All data accesses require explicit consent and are audited
- Policy store enforces least-privilege and purpose-bound data access
- System degrades gracefully when components are unavailable
- Model updates are staged; system reverts to rule-based on failure

---

## Citation

```bibtex
@article{aghealth2025,
  title={Agentic AI for Inclusive Nutrition and Healthcare},
  year={2025}
}
```
