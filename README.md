---
title: Project Blackout
emoji: "⚡"
colorFrom: blue
colorTo: indigo
sdk: docker
sdk_version: "1"
pinned: false
---

# ⚡ Project Blackout: Microgrid Power Dispatcher

> **OpenEnv Round 1 Submission** — High-stakes energy dispatching for critical infrastructure protection.

An AI agent manages a microgrid — balancing solar generation, battery storage, and flexible load shedding — to keep a **hospital (critical load) powered at all times** across three difficulty scenarios.

---

## 🎯 Motivation

Power grid management is a high-stakes real-world task that affects millions of lives. Modern microgrids with renewable sources (solar) and battery storage require intelligent dispatching agents that can:

- Predict and react to supply/demand imbalances in real time
- Protect critical loads (hospitals, emergency services) unconditionally
- Minimise energy waste and carbon footprint
- Operate under degraded conditions (grid failures, low battery, night-time)

This environment models exactly that — making it immediately useful for training and evaluating RL/LLM agents on consequential energy decisions.

---

## 🧩 Environment Architecture

```
┌─────────────────────────────────────────────────────┐
│                   PowerGridEnv                      │
│                                                     │
│  ☀️  Solar Array      →  time-varying generation    │
│  🔋 Battery (ESS)    →  charge / discharge          │
│  🏥 Hospital Load    →  CRITICAL — must always run  │
│  🏘️  Residential Load →  flexible — can be shed    │
│                                                     │
│  Power Balance:                                     │
│  net = (solar + battery_out) - (hospital + res)     │
└─────────────────────────────────────────────────────┘
```

---

## 📐 Observation Space

| Field          | Type  | Range            | Description                   |
| -------------- | ----- | ---------------- | ----------------------------- |
| `solar_mw`     | float | ≥ 0              | Current solar generation (MW) |
| `battery_soc`  | float | 0–1              | Battery state of charge       |
| `demand_total` | float | ≥ 0              | Total load demand (MW)        |
| `grid_status`  | enum  | NORMAL / FAILURE | Grid connectivity status      |

---

## 🎮 Action Space

| Field           | Type      | Description                                          |
| --------------- | --------- | ---------------------------------------------------- |
| `dispatch_type` | enum      | `CHARGE` / `DISCHARGE` / `SHED_RESIDENTIAL` / `NOOP` |
| `amount_mw`     | float ≥ 0 | Magnitude of action (MW), capped at 5.0 MW           |

### Action Semantics

- **CHARGE** — Store surplus solar energy in the battery (92% efficiency)
- **DISCHARGE** — Draw from battery to supply demand (94% efficiency)
- **SHED_RESIDENTIAL** — Drop residential load to reduce demand (penalty applied)
- **NOOP** — Take no action this step

---

## 🏆 Reward Function

| Event                          | Reward     |
| ------------------------------ | ---------- |
| Hospital powered (per step)    | **+10.0**  |
| Solar energy utilised (per MW) | +0.5       |
| Residential load shed (per MW) | −2.0       |
| Excess energy wasted (per MW)  | −1.0       |
| Battery throughput (per MW)    | −0.3       |
| Hospital blackout (terminal)   | **−100.0** |

The reward is **dense** — the agent receives signal every step, not just at episode end.

---

## 📋 Tasks

### Task 1 — Easy (`scenario: easy`)

- **Setup:** Solar peak 10 MW, hospital 2.5 MW, residential 3.5 MW, battery 85% SoC, hour 11 (peak sun)
- **Objective:** Keep hospital powered for all 10 steps
- **Scoring:** `steps_hospital_powered / 10`
- **Passing threshold:** ≥ 0.80
- **Expected baseline score:** ~1.0 (manageable with NOOP or simple discharge)

### Task 2 — Medium (`scenario: medium`)

- **Setup:** Solar peak 4 MW (post-sunset), hospital 2.0 MW, residential 3.5 MW, battery 70% SoC, hour 10
- **Objective:** Keep hospital alive AND minimise residential shedding
- **Scoring:** `0.6 × uptime + 0.4 × efficiency`
- **Passing threshold:** ≥ 0.60
- **Expected baseline score:** ~0.75–0.77

### Task 3 — Hard (`scenario: hard`)

- **Setup:** Zero solar, grid FAILURE, hospital 2.0 MW, residential 5.0 MW, battery 70% SoC, hour 2
- **Objective:** Survive as long as possible; shed ruthlessly to protect hospital
- **Scoring:** `0.7 × survival_ratio + 0.3 × shed_efficiency`
- **Passing threshold:** ≥ 0.30
- **Expected baseline score:** ~0.35–0.38

---

## 🚀 Setup & Usage

### Local Python

```bash
# Clone the repo
git clone https://huggingface.co/spaces/YOUR_USERNAME/project-blackout
cd project-blackout

# Install dependencies
pip install -r requirements.txt

# Run the FastAPI server
uvicorn server.app:app --host 0.0.0.0 --port 7860

# In another terminal — test it
curl http://localhost:7860/
curl -X POST http://localhost:7860/reset -H "Content-Type: application/json" -d '{"scenario":"easy"}'
```

### Docker

```bash
docker build -t project-blackout .
docker run -p 7860:7860 project-blackout

# Verify
curl http://localhost:7860/
```

### Run Inference (LLM agent)

```bash
export API_BASE_URL="https://your-endpoint/v1"
export MODEL_NAME="meta-llama/Llama-3-8b-instruct"
export HF_TOKEN="hf_..."

python inference.py --scenario easy
python inference.py --scenario medium
python inference.py --scenario hard
```

### Run Task Graders Directly

```bash
python tasks/task_easy.py
python tasks/task_medium.py
python tasks/task_hard.py
```

---

## 📡 API Reference

| Method | Endpoint            | Description                                                           |
| ------ | ------------------- | --------------------------------------------------------------------- |
| GET    | `/`                 | Health check                                                          |
| POST   | `/reset`            | Reset env (`{"scenario": "easy"}`)                                    |
| POST   | `/step`             | Step with action (`{"dispatch_type": "DISCHARGE", "amount_mw": 3.0}`) |
| GET    | `/state`            | Internal diagnostics                                                  |
| GET    | `/grade`            | Current episode score                                                 |
| GET    | `/tasks`            | List all tasks                                                        |
| POST   | `/tasks/{id}/run`   | Run a full task with default agent                                    |
| GET    | `/tasks/{id}/grade` | Grade current episode with task grader                                |

---

## 📊 Baseline Scores

| Task   | Score | Threshold | Status  |
| ------ | ----- | --------- | ------- |
| easy   | ~1.00 | 0.80      | ✅ PASS |
| medium | ~0.76 | 0.60      | ✅ PASS |
| hard   | ~0.37 | 0.30      | ✅ PASS |

_Scores obtained with the built-in default agents. LLM agent scores vary by model._

---

## 🗂️ Project Structure

```
project-blackout/
├── openenv.yaml        # OpenEnv metadata & task registry
├── models.py           # Pydantic typed models (Observation, Action, Reward)
├── env.py              # PowerGridEnv simulation logic
├── grader.py           # Generic episode grader
├── inference.py        # LLM inference loop (OpenAI-compatible)
├── main.py             # FastAPI server
├── requirements.txt    # Python dependencies
├── Dockerfile          # Container definition
├── tasks/
│   ├── __init__.py
│   ├── task_easy.py    # Easy task + grader
│   ├── task_medium.py  # Medium task + grader
│   └── task_hard.py    # Hard task + grader
└── README.md
```

---

## ⚙️ Environment Variables

| Variable       | Description                             |
| -------------- | --------------------------------------- |
| `API_BASE_URL` | OpenAI-compatible API base URL          |
| `MODEL_NAME`   | Model identifier for inference          |
| `HF_TOKEN`     | Hugging Face / API authentication token |

---

## 🔬 Design Notes

- **Stochastic solar:** Gaussian noise (±5%) added to solar output for realism
- **Time-of-day demand:** 24-hour residential demand profile — peaks at 17:00, troughs at 04:00
- **Battery degradation:** tracked via throughput (cosmetic penalty, bonus feature)
- **No GUI/Matplotlib:** fully headless, API-only, runs on 2 vCPU / 8 GB RAM
- **Episode length:** 10 steps (each step = 1 hour of simulated time)

---

## 📄 License

MIT License — built for the OpenEnv Hackathon, Round 1.
