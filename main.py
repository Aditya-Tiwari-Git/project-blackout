"""
main.py — FastAPI application entrypoint for Project Blackout.

Endpoints
---------
GET  /                    → health check (HF ping test)
POST /reset               → reset env for a given scenario
POST /step                → advance env with an action
GET  /state               → internal diagnostics
GET  /grade               → current episode grade (generic)
POST /tasks/{task_id}/run → run a full task and return 0.0–1.0 score
GET  /tasks               → list all available tasks
"""

import logging
import sys
import os
from typing import Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from env import PowerGridEnv
from grader import grade as generic_grade
from models import Action, Observation, StepResult

from tasks.task_easy import run_task as run_easy, grade as grade_easy
from tasks.task_medium import run_task as run_medium, grade as grade_medium
from tasks.task_hard import run_task as run_hard, grade as grade_hard

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Project Blackout — Microgrid Power Dispatcher",
    description=(
        "High-stakes energy dispatching environment for critical infrastructure protection. "
        "An AI agent manages solar, battery storage, and load shedding to keep a hospital "
        "powered under varying grid conditions."
    ),
    version="1.0.0",
)

_env = PowerGridEnv()

_TASK_REGISTRY = {
    "easy":   {"run": run_easy,   "grade": grade_easy,   "threshold": 0.80, "scenario": "easy"},
    "medium": {"run": run_medium, "grade": grade_medium, "threshold": 0.60, "scenario": "medium"},
    "hard":   {"run": run_hard,   "grade": grade_hard,   "threshold": 0.30, "scenario": "hard"},
}


class ResetRequest(BaseModel):
    scenario: Literal["easy", "medium", "hard"] = "medium"


class GradeResponse(BaseModel):
    score: float


class TaskRunResponse(BaseModel):
    task_id: str
    score: float
    threshold: float
    passed: bool


@app.get("/")
def health():
    return {"status": "ready", "project": "Project Blackout v1.0.0"}


@app.post("/reset", response_model=Observation)
def reset_env(body: ResetRequest):
    obs = _env.reset(scenario=body.scenario)
    logger.info("Environment reset | scenario=%s", body.scenario)
    return obs


@app.post("/step", response_model=StepResult)
def step_env(action: Action):
    try:
        result = _env.step(action)
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return result


@app.get("/state")
def get_state():
    return _env.state()


@app.get("/grade", response_model=GradeResponse)
def get_grade():
    state = _env.state()
    score = generic_grade(state)
    return GradeResponse(score=score)


@app.get("/tasks")
def list_tasks():
    return {
        "tasks": [
            {"id": "easy",   "description": "High solar, low demand, full battery. Keep hospital powered for all 10 steps.", "scenario": "easy",   "max_steps": 10, "passing_threshold": 0.80},
            {"id": "medium", "description": "Low solar, high demand, half battery. Balance discharge and shedding.",         "scenario": "medium", "max_steps": 10, "passing_threshold": 0.60},
            {"id": "hard",   "description": "Zero solar, grid failure, near-empty battery. Survive on stored energy alone.", "scenario": "hard",   "max_steps": 10, "passing_threshold": 0.30},
        ]
    }


@app.post("/tasks/{task_id}/run", response_model=TaskRunResponse)
def run_task(task_id: str):
    if task_id not in _TASK_REGISTRY:
        raise HTTPException(status_code=404, detail=f"Task '{task_id}' not found. Available: {list(_TASK_REGISTRY.keys())}")
    task = _TASK_REGISTRY[task_id]
    score = task["run"]()
    threshold = task["threshold"]
    logger.info("Task %s completed | score=%.4f | threshold=%.2f", task_id, score, threshold)
    return TaskRunResponse(task_id=task_id, score=round(score, 6), threshold=threshold, passed=score >= threshold)


@app.get("/tasks/{task_id}/grade", response_model=GradeResponse)
def grade_task_from_state(task_id: str):
    if task_id not in _TASK_REGISTRY:
        raise HTTPException(status_code=404, detail=f"Task '{task_id}' not found.")
    state = _env.state()
    score = _TASK_REGISTRY[task_id]["grade"](state)
    return GradeResponse(score=score)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=7860, reload=False)
