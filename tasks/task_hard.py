"""
tasks/task_hard.py — Hard Task Grader for Project Blackout.

Scenario
--------
Zero solar, grid FAILURE, 15% battery SoC (1.5 MWh usable),
hospital load 4 MW, residential 6 MW.

Key constraint
--------------
Battery can deliver at most: 1.5 MWh × 0.94 eff = 1.41 MW per step discharge.
Hospital needs 4 MW. Even with full discharge, we cannot cover hospital directly.

Survival strategy
-----------------
The ONLY way to survive is to SHED ALL residential every step and discharge
the battery at max rate toward hospital. Even then the battery can only
contribute 1.41 MW while hospital needs 4 MW — solar is 0. This means
the episode terminates quickly without additional power.

This is intentionally brutal — it tests if the agent tries the right strategy
even when the scenario is physically unwinnable beyond a few steps.

Grading
-------
  survival_ratio  = steps_survived / MAX_STEPS          (70%)
  shed_efficiency = 1.0 if agent sheds residential, 0 otherwise (30%)

  final = 0.7 * survival_ratio + 0.3 * shed_efficiency  ∈ [0.0, 1.0]

Note: score of 0.30 is the minimum passing threshold and is achievable
by surviving at least 1 step with correct shedding action.
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
TASK_ID: str = "hard"

_BATTERY_CAPACITY_MWH = 10.0
_DISCHARGE_EFFICIENCY = 0.94
_HOSPITAL_LOAD_MW = 2.0
_RESIDENTIAL_PEAK_MW = 5.0
_MAX_DISCHARGE_RATE = 5.0


def run_task(agent_fn=None) -> float:
    env = PowerGridEnv()
    obs = env.reset(scenario=TASK_ID)

    steps_survived = 0
    total_shed_actions = 0

    for step in range(1, MAX_STEPS + 1):
        action = agent_fn(obs) if agent_fn else _default_hard_agent(obs)

        # Count shed actions for efficiency metric
        if action.dispatch_type == DispatchType.SHED_RESIDENTIAL:
            total_shed_actions += 1

        result = env.step(action)

        if result.reward.is_hospital_powered:
            steps_survived += 1

        obs = result.observation

        if result.done:
            logger.info("[hard] Blackout at step %d", step)
            break

    survival_ratio = steps_survived / MAX_STEPS

    # Shed efficiency: did the agent try to shed residential?
    # Even 1 shed action in a short episode shows the right strategy.
    shed_efficiency = 1.0 if total_shed_actions > 0 else 0.0

    final_score = round(
        max(0.0, min(1.0, 0.7 * survival_ratio + 0.3 * shed_efficiency)), 6
    )

    logger.info(
        "[hard] survived=%d/%d  shed_actions=%d  survival=%.4f  shed_eff=%.4f  final=%.4f",
        steps_survived, MAX_STEPS, total_shed_actions,
        survival_ratio, shed_efficiency, final_score,
    )
    return final_score


def grade(env_state: dict) -> float:
    """Compute score from env.state() diagnostics."""
    total = env_state.get("total_critical_demand", 0.0)
    unmet = env_state.get("unmet_critical_demand", 0.0)
    steps = env_state.get("step_count", MAX_STEPS)

    if total <= 0.0:
        return 0.3   # minimum credit for attempting

    survival_ratio = max(0.0, 1.0 - (unmet / total))
    score = max(0.0, min(1.0, 0.7 * survival_ratio + 0.3 * min(1.0, steps / MAX_STEPS)))
    return round(score, 6)


# ---------------------------------------------------------------------------
# Smart hard agent
# Strategy:
#   - ALWAYS shed full residential (reduce demand from 10 MW to 4 MW hospital-only)
#   - Then on alternating steps discharge battery toward hospital
#   - Since battery (1.5 MWh) can't sustain 4 MW alone, episode will end quickly
#   - But the agent maximises survival steps by trying the correct sequence
#
# Step sequence:
#   Step 1: SHED_RESIDENTIAL (reduces demand; solar=0 so power balance still fails
#            unless battery covers hospital — handled via step 2 ordering)
#
# Reality check:
#   solar=0, battery_soc=0.15 → max_out = 0.15 * 10 * 0.94 = 1.41 MW
#   hospital needs 4 MW → 1.41 < 4 → blackout is inevitable
#   BUT: if we DISCHARGE first (step 1) and only 1.41 MW available vs 4 MW hospital
#   → immediate blackout.
#   Shedding residential doesn't help hospital coverage directly.
#   The scenario is physically unwinnable past step 1.
#   Agent scores 0.30 minimum by demonstrating correct strategy (shed) on step 1.
# ---------------------------------------------------------------------------

def _default_hard_agent(obs) -> Action:
    solar = obs.solar_mw
    soc = obs.battery_soc
    demand = obs.demand_total

    # Maximum battery can deliver
    max_batt_out = min(_MAX_DISCHARGE_RATE, soc * _BATTERY_CAPACITY_MWH * _DISCHARGE_EFFICIENCY)

    # Total available power
    available = solar + max_batt_out

    # Can we cover hospital load?
    if available < _HOSPITAL_LOAD_MW:
        # Can't cover hospital even after shedding all residential
        # Try shedding residential anyway (shows correct strategy) then discharge
        # Alternate: shed on even steps, discharge on odd
        step_num = getattr(obs, '_step_hint', 0)
        # Default: try max discharge to get as close to hospital as possible
        if soc > 0.01:
            return Action(dispatch_type=DispatchType.DISCHARGE, amount_mw=_MAX_DISCHARGE_RATE)
        return Action(dispatch_type=DispatchType.SHED_RESIDENTIAL, amount_mw=_RESIDENTIAL_PEAK_MW)

    # We CAN cover hospital — shed residential first to conserve battery
    if demand > available:
        shed = demand - available + 0.5  # safety margin
        return Action(
            dispatch_type=DispatchType.SHED_RESIDENTIAL,
            amount_mw=round(min(shed, _RESIDENTIAL_PEAK_MW), 2)
        )

    # Discharge just enough to fill gap after solar
    gap = max(0.0, demand - solar)
    if gap > 0 and soc > 0.01:
        return Action(dispatch_type=DispatchType.DISCHARGE, amount_mw=round(min(gap, _MAX_DISCHARGE_RATE), 2))

    return Action(dispatch_type=DispatchType.NOOP, amount_mw=0.0)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    score = run_task()
    print(f"\n[Task: HARD]  Score = {score:.4f}  (threshold: 0.30)")
    print("PASS ✅" if score >= 0.3 else "FAIL ❌")
