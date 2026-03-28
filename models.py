"""
models.py — Strictly typed Pydantic models for Project Blackout: Microgrid Power Dispatcher
"""

from enum import Enum
from pydantic import BaseModel, Field, validator


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class GridStatus(str, Enum):
    NORMAL = "NORMAL"
    FAILURE = "FAILURE"


class DispatchType(str, Enum):
    CHARGE = "CHARGE"
    DISCHARGE = "DISCHARGE"
    SHED_RESIDENTIAL = "SHED_RESIDENTIAL"
    NOOP = "NOOP"


# ---------------------------------------------------------------------------
# Core Models
# ---------------------------------------------------------------------------

class Observation(BaseModel):
    """Observation returned to the agent after each step."""

    solar_mw: float = Field(..., description="Current solar generation in MW (≥ 0)")
    battery_soc: float = Field(
        ..., ge=0.0, le=1.0, description="Battery state of charge (0 = empty, 1 = full)"
    )
    demand_total: float = Field(..., description="Total load demand in MW")
    grid_status: GridStatus = Field(..., description="Current grid connectivity status")

    class Config:
        use_enum_values = True


class Action(BaseModel):
    """Action submitted by the agent each step."""

    dispatch_type: DispatchType = Field(..., description="Type of dispatch action")
    amount_mw: float = Field(
        ..., ge=0.0, description="Magnitude of action in MW (must be non-negative)"
    )

    @validator("amount_mw")
    def clamp_amount(cls, v: float) -> float:  # noqa: N805
        """Guard against accidental negative values from LLM output."""
        return max(0.0, v)

    class Config:
        use_enum_values = True


class Reward(BaseModel):
    """Reward signal returned after each environment step."""

    score: float = Field(..., description="Scalar reward for this timestep")
    is_hospital_powered: bool = Field(
        ..., description="True if the critical hospital load was fully met"
    )


class StepResult(BaseModel):
    """Full result bundle returned by env.step()."""

    observation: Observation
    reward: Reward
    done: bool = Field(..., description="True if the episode has ended")
    info: dict = Field(default_factory=dict, description="Auxiliary diagnostics")
