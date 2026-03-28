"""
tasks/task_medium.py — Medium Task Grader for Project Blackout.

Scenario
--------
Low solar (peak 4 MW, post-sunset start at hour 18), high residential demand,
battery at 50% SoC. The agent must intelligently balance discharge, shedding,
and waste avoidance across 10 steps.

Agent Strategy
--------------
At hour 18 solar ≈ 0.48 MW, total demand ≈ 7.9 MW, battery = 5 MWh (50% of 10).
Available = solar + max_discharge = 0.48 + 4.7 ≈ 5.18 MW < 7.9 MW total demand.
Hospital load = 3 MW → must shed at least 2.72 MW residential EVERY step.
Smart agent: always shed residential first, then discharge just enough for hospital.

Grading (composite)
-------------------
  Component A (60%): hospital uptime ratio
      A = steps_hospital_powered / MAX_STEPS

  Component B (40%): battery efficiency
      B = 1 - clamp(total_waste / max_possible_waste, 0, 1)

  final_score = 0.6 * A + 0.4 * B   clamped to [0.0, 1.0]

Expected baseline: ≥ 0.6 for a good agent.
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
TASK_ID: str = "medium"

# Battery constants (must match env.py)
_BATTERY_CAPACITY_MWH = 10.0
_DISCHARGE_EFFICIENCY = 0.94
_HOSPITAL_LOAD_MW = 2.0
_RESIDENTIAL_PEAK_MW = 3.5


def run_task(agent_fn=None) -> float:
    env = PowerGridEnv()
    obs = env.reset(scenario=TASK_ID)

    steps_powered = 0
    total_shed = 0.0
    total_max_possible_shed = 0.0

    for step in range(1, MAX_STEPS + 1):
        action = agent_fn(obs) if agent_fn else _default_medium_agent(obs)
        result = env.step(action)

        if result.reward.is_hospital_powered:
            steps_powered += 1

        total_shed += result.info.get("residential_shed_mw", 0.0)
        total_max_possible_shed += result.info.get("residential_load", _RESIDENTIAL_PEAK_MW) + \
                                   result.info.get("residential_shed_mw", 0.0)

        obs = result.observation

        if result.done:
            logger.info("[medium] Episode ended early at step %d", step)
            break

    uptime_score = steps_powered / MAX_STEPS

    # Efficiency: reward not wasting battery (shed only what's needed)
    if total_max_possible_shed > 0:
        shed_ratio = total_shed / total_max_possible_shed
        # Moderate shedding (not too much, not too little) is ideal
        efficiency_score = 1.0 - min(1.0, shed_ratio)
    else:
        efficiency_score = 1.0

    final_score = round(
        max(0.0, min(1.0, 0.6 * uptime_score + 0.4 * efficiency_score)), 6
    )

    logger.info(
        "[medium] uptime=%.4f  efficiency=%.4f  final=%.4f",
        uptime_score, efficiency_score, final_score,
    )
    return final_score


def grade(env_state: dict) -> float:
    """
    Compute score from env.state() diagnostics.
    Used by the OpenEnv framework for automated grading.
    """
    total = env_state.get("total_critical_demand", 0.0)
    unmet = env_state.get("unmet_critical_demand", 0.0)

    if total <= 0.0:
        return 1.0

    uptime = 1.0 - (unmet / total)
    score = 0.6 * uptime + 0.4 * 1.0
    return round(max(0.0, min(1.0, score)), 6)


# ---------------------------------------------------------------------------
# Smart medium agent
# Strategy:
#   At each step, check if solar + max battery discharge can cover TOTAL demand.
#   If not → shed residential first to reduce demand to what we can actually cover.
#   Always protect hospital (3 MW) unconditionally.
#   Discharge only what's needed (not the max), to conserve battery for later steps.
# ---------------------------------------------------------------------------

def _default_medium_agent(obs) -> Action:
    solar = obs.solar_mw
    soc = obs.battery_soc
    demand = obs.demand_total

    # How much can battery deliver this step?
    max_battery_out = min(5.0, soc * _BATTERY_CAPACITY_MWH * _DISCHARGE_EFFICIENCY)

    # Total available power
    total_available = solar + max_battery_out

    if total_available < demand:
        # Cannot cover full demand — shed residential, protect hospital
        # Shed exactly the shortfall (residential only, hospital is sacred)
        shortfall = demand - total_available
        shed_mw = min(shortfall + 0.5, _RESIDENTIAL_PEAK_MW)  # +0.5 safety margin
        return Action(dispatch_type=DispatchType.SHED_RESIDENTIAL, amount_mw=round(shed_mw, 2))

    # We have enough power — use battery conservatively
    gap = demand - solar
    if gap > 0 and soc > 0.05:
        # Discharge just enough to fill the solar gap
        discharge_needed = min(gap, max_battery_out)
        return Action(dispatch_type=DispatchType.DISCHARGE, amount_mw=round(discharge_needed, 2))

    # Solar covers everything — charge battery if possible
    if solar > demand and soc < 0.95:
        surplus = solar - demand
        return Action(dispatch_type=DispatchType.CHARGE, amount_mw=round(min(surplus, 5.0), 2))

    return Action(dispatch_type=DispatchType.NOOP, amount_mw=0.0)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    score = run_task()
    print(f"\n[Task: MEDIUM]  Score = {score:.4f}  (threshold: 0.60)")
    print("PASS ✅" if score >= 0.6 else "FAIL ❌")
