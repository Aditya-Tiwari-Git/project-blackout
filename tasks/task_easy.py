"""
tasks/task_easy.py — Easy Task Grader for Project Blackout.

Scenario
--------
High solar output (peak 10 MW), low demand, battery starts at 85% SoC.
The agent simply needs to avoid a blackout for all MAX_STEPS steps.

Grading
-------
score = steps_hospital_powered / MAX_STEPS

  1.0  → hospital powered every single step (perfect)
  0.0  → hospital blacked out on step 1
  partial → proportional to uptime

Expected baseline: a competent agent should score ≥ 0.8.
"""

import sys
import os
import logging

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env import PowerGridEnv
from models import Action, DispatchType

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

MAX_STEPS: int = 10
TASK_ID: str = "easy"


def run_task(agent_fn=None) -> float:
    """
    Run the easy task and return a score in [0.0, 1.0].

    Parameters
    ----------
    agent_fn : callable, optional
        A function that takes an Observation and returns an Action.
        Defaults to a simple NOOP agent (still scores well on easy).

    Returns
    -------
    float
        Normalised score in [0.0, 1.0].
    """
    env = PowerGridEnv()
    obs = env.reset(scenario=TASK_ID)

    steps_powered = 0

    for step in range(1, MAX_STEPS + 1):
        # Use provided agent or fallback to a solar-aware default
        if agent_fn is not None:
            action = agent_fn(obs)
        else:
            action = _default_easy_agent(obs)

        result = env.step(action)

        if result.reward.is_hospital_powered:
            steps_powered += 1

        obs = result.observation

        if result.done:
            logger.info("[easy] Episode ended early at step %d", step)
            break

    score = round(steps_powered / MAX_STEPS, 6)
    logger.info("[easy] steps_powered=%d / %d → score=%.4f", steps_powered, MAX_STEPS, score)
    return score


def grade(env_state: dict) -> float:
    """
    Compute score from env.state() diagnostics.
    Used by the OpenEnv framework for automated grading.
    """
    total = env_state.get("total_critical_demand", 0.0)
    unmet = env_state.get("unmet_critical_demand", 0.0)

    if total <= 0.0:
        return 1.0

    score = 1.0 - (unmet / total)
    return round(max(0.0, min(1.0, score)), 6)


# ---------------------------------------------------------------------------
# Default easy agent: discharge when demand > solar, else NOOP
# ---------------------------------------------------------------------------

def _default_easy_agent(obs) -> Action:
    gap = obs.demand_total - obs.solar_mw
    if gap > 0 and obs.battery_soc > 0.1:
        return Action(dispatch_type=DispatchType.DISCHARGE, amount_mw=min(gap, 5.0))
    elif obs.solar_mw > obs.demand_total and obs.battery_soc < 0.95:
        surplus = obs.solar_mw - obs.demand_total
        return Action(dispatch_type=DispatchType.CHARGE, amount_mw=min(surplus, 5.0))
    return Action(dispatch_type=DispatchType.NOOP, amount_mw=0.0)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    score = run_task()
    print(f"\n[Task: EASY]  Score = {score:.4f}  (threshold: 0.80)")
    print("PASS ✅" if score >= 0.8 else "FAIL ❌")
