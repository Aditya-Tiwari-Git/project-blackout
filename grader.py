"""
grader.py — Scoring module for Project Blackout: Microgrid Power Dispatcher.

Score Formula
-------------
    score = 1.0 - (unmet_critical_demand / total_critical_demand)

  • Clamped to [0.0, 1.0].
  • 1.0 = perfect run — hospital always powered.
  • 0.0 = catastrophic failure — hospital never powered.
"""

import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)


def grade(env_state: Dict[str, Any]) -> float:
    """
    Compute a normalised episode score from environment diagnostics.

    Parameters
    ----------
    env_state : dict
        Dictionary returned by ``PowerGridEnv.state()``.
        Expected keys:
          - ``total_critical_demand``  (float) : cumulative hospital MWh demanded
          - ``unmet_critical_demand``  (float) : cumulative hospital MWh unmet

    Returns
    -------
    float
        Score in [0.0, 1.0].
    """
    total: float = env_state.get("total_critical_demand", 0.0)
    unmet: float = env_state.get("unmet_critical_demand", 0.0)

    if total <= 0.0:
        # No critical demand was ever presented — treat as perfect (edge case)
        logger.warning("[grader] total_critical_demand is zero; returning 1.0 by default.")
        return 1.0

    raw_score = 1.0 - (unmet / total)
    score = max(0.0, min(1.0, raw_score))   # clamp to [0, 1]

    logger.info(
        "[grader] total_demand=%.4f  unmet=%.4f  raw_score=%.6f  final_score=%.4f",
        total, unmet, raw_score, score,
    )
    return round(score, 6)


# ---------------------------------------------------------------------------
# CLI convenience
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import json
    import sys

    if len(sys.argv) < 2:
        print("Usage: python grader.py '<json_state>'")
        sys.exit(1)

    state_dict = json.loads(sys.argv[1])
    result = grade(state_dict)
    print(f"Grade: {result:.4f}")
