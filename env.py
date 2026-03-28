"""
env.py — PowerGridEnv: Microgrid simulation environment for Project Blackout.

Scenarios
---------
easy   : High solar, low demand, full battery
medium : Low solar, high demand, half battery
hard   : Zero solar, high demand, grid failure, near-empty battery

Power Balance (per step)
------------------------
    net = (solar + battery_discharge) - (hospital_load + residential_load)

Reward shaping
--------------
  +10   hospital powered
  +0.5  per MWh of solar utilised (CO₂ proxy)
  -5    hospital NOT powered  (episode also terminates with -100 terminal penalty)
  -2    per MW of residential load shed
  -1    per MW of excess / wasted energy
  -0.3  per charge/discharge cycle (battery wear proxy)
"""

import logging
import math
import random
from typing import Dict, Any

from models import (
    Action,
    DispatchType,
    GridStatus,
    Observation,
    Reward,
    StepResult,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BATTERY_CAPACITY_MWH: float = 10.0        # maximum storable energy
BATTERY_MAX_RATE_MW: float = 5.0          # max charge / discharge per step
CHARGE_EFFICIENCY: float = 0.92           # energy retained when charging
DISCHARGE_EFFICIENCY: float = 0.94        # energy delivered when discharging
BATTERY_DEGRADATION_PER_CYCLE: float = 0.0005  # SoC penalty per full cycle (bonus feature)

HOSPITAL_PENALTY: float = -100.0          # terminal penalty for blackout
STEP_BLACKOUT_PENALTY: float = -5.0       # per-step pre-terminal penalty signal
RESIDENTIAL_SHED_PENALTY: float = -2.0    # per MW shed
WASTE_PENALTY: float = -1.0               # per MW wasted
SOLAR_BONUS: float = 0.5                  # per MW of solar used
HOSPITAL_POWERED_BONUS: float = 10.0      # per step hospital stays live
BATTERY_CYCLE_PENALTY: float = -0.3       # per MW moved through battery

# Solar profile: 24-hour normalised generation fraction (peaks at ~13:00)
_SOLAR_BASE_PROFILE: list = [
    0.00, 0.00, 0.00, 0.00, 0.00, 0.02,  # 00–05
    0.08, 0.20, 0.40, 0.60, 0.78, 0.90,  # 06–11
    0.95, 0.92, 0.85, 0.70, 0.52, 0.30,  # 12–17
    0.12, 0.04, 0.01, 0.00, 0.00, 0.00,  # 18–23
]

# Demand multipliers: residential load varies through the day
_RESIDENTIAL_DEMAND_PROFILE: list = [
    0.55, 0.50, 0.48, 0.47, 0.48, 0.52,  # 00–05 (night trough)
    0.60, 0.72, 0.85, 0.90, 0.88, 0.85,  # 06–11 (morning ramp)
    0.82, 0.80, 0.82, 0.88, 0.95, 1.00,  # 12–17 (afternoon peak)
    0.98, 0.92, 0.85, 0.75, 0.65, 0.58,  # 18–23 (evening)
]


# ---------------------------------------------------------------------------
# PowerGridEnv
# ---------------------------------------------------------------------------

class PowerGridEnv:
    """OpenEnv-compliant microgrid power dispatching environment."""

    def __init__(self) -> None:
        self._battery_soc: float = 0.5
        self._battery_capacity: float = BATTERY_CAPACITY_MWH
        self._current_hour: int = 0
        self._solar_peak_mw: float = 8.0          # scenario-dependent
        self._hospital_load_mw: float = 3.0        # constant critical load
        self._residential_peak_mw: float = 5.0     # scenario-dependent
        self._grid_status: GridStatus = GridStatus.NORMAL
        self._total_co2_saved: float = 0.0
        self._total_critical_demand: float = 0.0
        self._unmet_critical_demand: float = 0.0
        self._step_count: int = 0
        self._done: bool = False
        self._stochastic: bool = True              # enable solar noise (bonus)
        self._total_battery_throughput: float = 0.0  # for degradation tracking

    # ------------------------------------------------------------------
    # reset
    # ------------------------------------------------------------------

    def reset(self, scenario: str = "medium") -> Observation:
        """
        Reset the environment for a new episode.

        Parameters
        ----------
        scenario : str
            One of "easy", "medium", "hard".
        """
        scenario = scenario.lower().strip()
        self._step_count = 0
        self._done = False
        self._total_co2_saved = 0.0
        self._total_critical_demand = 0.0
        self._unmet_critical_demand = 0.0
        self._total_battery_throughput = 0.0
        self._grid_status = GridStatus.NORMAL

        if scenario == "easy":
            self._battery_soc = 0.85
            self._current_hour = 11          # peak solar hour
            self._solar_peak_mw = 10.0
            self._hospital_load_mw = 2.5
            self._residential_peak_mw = 3.5
            logger.info("[reset] Scenario: EASY — high solar, low demand")

        elif scenario == "hard":
            self._battery_soc = 0.70
            self._current_hour = 2           # dead of night
            self._solar_peak_mw = 0.0        # no solar
            self._hospital_load_mw = 2.0
            self._residential_peak_mw = 5.0
            self._grid_status = GridStatus.FAILURE
            logger.info("[reset] Scenario: HARD — no solar, grid failure, low battery")

        else:  # medium (default)
            self._battery_soc = 0.70
            self._current_hour = 10          # mid-morning, solar declining after noon
            self._solar_peak_mw = 4.0
            self._hospital_load_mw = 2.0
            self._residential_peak_mw = 3.5
            logger.info("[reset] Scenario: MEDIUM — moderate solar, battery-dependent from dusk")

        return self._build_observation()

    # ------------------------------------------------------------------
    # step
    # ------------------------------------------------------------------

    def step(self, action: Action) -> StepResult:
        """
        Advance the environment by one timestep.

        Parameters
        ----------
        action : Action
            The dispatch action chosen by the agent.

        Returns
        -------
        StepResult
            Next observation, reward, done flag, and diagnostics.
        """
        if self._done:
            raise RuntimeError("Episode is done — call reset() before stepping.")

        self._step_count += 1
        hour = self._current_hour

        # ---- Solar generation (with optional stochastic noise) --------
        solar_fraction = _SOLAR_BASE_PROFILE[hour % 24]
        solar_mw = self._solar_peak_mw * solar_fraction
        if self._stochastic and solar_mw > 0:
            noise = random.gauss(0, 0.05)          # ±5% Gaussian noise
            solar_mw = max(0.0, solar_mw * (1 + noise))

        # ---- Demand (time-of-day variation) ---------------------------
        res_fraction = _RESIDENTIAL_DEMAND_PROFILE[hour % 24]
        hospital_load = self._hospital_load_mw
        residential_load = self._residential_peak_mw * res_fraction

        # ---- Apply action ---------------------------------------------
        battery_delta_mw = 0.0          # net MW into (+) or out of (-) battery
        residential_shed_mw = 0.0
        battery_throughput = 0.0

        dispatch = action.dispatch_type
        amount = min(action.amount_mw, BATTERY_MAX_RATE_MW)  # respect rate limit

        if dispatch == DispatchType.CHARGE:
            # Store solar surplus → battery
            effective_charge = amount * CHARGE_EFFICIENCY
            soc_increase = effective_charge / self._battery_capacity
            actual_soc_increase = min(soc_increase, 1.0 - self._battery_soc)
            self._battery_soc = min(1.0, self._battery_soc + actual_soc_increase)
            actual_charge_mw = actual_soc_increase * self._battery_capacity / CHARGE_EFFICIENCY
            battery_delta_mw = -actual_charge_mw   # consumed from grid
            battery_throughput = actual_charge_mw

        elif dispatch == DispatchType.DISCHARGE:
            # Discharge battery to supply load
            soc_decrease = (amount / DISCHARGE_EFFICIENCY) / self._battery_capacity
            actual_soc_decrease = min(soc_decrease, self._battery_soc)
            self._battery_soc = max(0.0, self._battery_soc - actual_soc_decrease)
            delivered_mw = actual_soc_decrease * self._battery_capacity * DISCHARGE_EFFICIENCY
            battery_delta_mw = delivered_mw     # injected to grid
            battery_throughput = delivered_mw

        elif dispatch == DispatchType.SHED_RESIDENTIAL:
            residential_shed_mw = min(amount, residential_load)
            residential_load -= residential_shed_mw

        # NOOP: no battery action, no shedding

        # Battery wear (degradation bonus feature)
        self._total_battery_throughput += battery_throughput

        # ---- Power balance --------------------------------------------
        available_power = solar_mw + battery_delta_mw   # net available from generation side
        total_load = hospital_load + residential_load
        net_balance = available_power - total_load

        # ---- Critical constraint check --------------------------------
        hospital_powered = True
        if available_power < hospital_load:
            # Hospital demand unmet → episode ends immediately
            hospital_powered = False
            unmet = hospital_load - available_power
            self._unmet_critical_demand += unmet
            self._total_critical_demand += hospital_load

            reward = Reward(score=HOSPITAL_PENALTY, is_hospital_powered=False)
            self._done = True
            obs = self._build_observation(solar_mw=solar_mw, demand_total=total_load)
            logger.warning(
                "[step %d] BLACKOUT — hospital unmet by %.2f MW. Episode terminated.",
                self._step_count, unmet,
            )
            return StepResult(
                observation=obs,
                reward=reward,
                done=True,
                info={
                    "hospital_load": hospital_load,
                    "residential_load": residential_load,
                    "solar_mw": solar_mw,
                    "net_balance": net_balance,
                    "blackout": True,
                },
            )

        # Hospital is powered; accumulate demand tracking
        self._total_critical_demand += hospital_load

        # ---- Solar CO₂ savings proxy ----------------------------------
        solar_utilised = min(solar_mw, total_load)
        self._total_co2_saved += solar_utilised * 0.4   # 0.4 tCO₂/MWh avoided

        # ---- Reward shaping -------------------------------------------
        r = HOSPITAL_POWERED_BONUS                      # baseline: hospital alive
        r += solar_utilised * SOLAR_BONUS               # solar usage reward
        r += residential_shed_mw * RESIDENTIAL_SHED_PENALTY  # shed penalty
        r += battery_throughput * BATTERY_CYCLE_PENALTY      # battery wear

        if net_balance > 0:
            # Excess energy wasted (no export modelled)
            waste_mw = net_balance
            r += waste_mw * WASTE_PENALTY

        # ---- Advance time ---------------------------------------------
        self._current_hour = (self._current_hour + 1) % 24

        obs = self._build_observation(solar_mw=solar_mw, demand_total=total_load)
        reward = Reward(score=round(r, 4), is_hospital_powered=True)

        logger.info(
            "[step %d] hour=%d solar=%.2f batt_soc=%.3f demand=%.2f shed=%.2f reward=%.3f",
            self._step_count, hour, solar_mw, self._battery_soc,
            total_load, residential_shed_mw, r,
        )

        return StepResult(observation=obs, reward=reward, done=False, info={
            "hospital_load": hospital_load,
            "residential_load": residential_load,
            "solar_mw": solar_mw,
            "net_balance": net_balance,
            "solar_utilised": solar_utilised,
            "residential_shed_mw": residential_shed_mw,
            "battery_throughput_mw": battery_throughput,
            "co2_saved_total": round(self._total_co2_saved, 4),
        })

    # ------------------------------------------------------------------
    # state
    # ------------------------------------------------------------------

    def state(self) -> Dict[str, Any]:
        """Return internal diagnostics (not part of agent observation)."""
        return {
            "battery_capacity": self._battery_capacity,
            "current_hour": self._current_hour,
            "total_co2_saved": round(self._total_co2_saved, 4),
            "total_battery_throughput_mw": round(self._total_battery_throughput, 4),
            "battery_degradation_fraction": round(
                self._total_battery_throughput * BATTERY_DEGRADATION_PER_CYCLE, 6
            ),
            "step_count": self._step_count,
            "unmet_critical_demand": round(self._unmet_critical_demand, 4),
            "total_critical_demand": round(self._total_critical_demand, 4),
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_observation(
        self,
        solar_mw: float | None = None,
        demand_total: float | None = None,
    ) -> Observation:
        """Construct an Observation from current env state."""
        if solar_mw is None:
            hour = self._current_hour
            frac = _SOLAR_BASE_PROFILE[hour % 24]
            solar_mw = self._solar_peak_mw * frac

        if demand_total is None:
            hour = self._current_hour
            res_frac = _RESIDENTIAL_DEMAND_PROFILE[hour % 24]
            demand_total = self._hospital_load_mw + self._residential_peak_mw * res_frac

        return Observation(
            solar_mw=round(solar_mw, 4),
            battery_soc=round(self._battery_soc, 6),
            demand_total=round(demand_total, 4),
            grid_status=self._grid_status,
        )
