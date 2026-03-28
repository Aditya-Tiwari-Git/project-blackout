"""
tasks/ — Task definitions and graders for Project Blackout.

Available tasks
---------------
task_easy   : High solar, low demand. Threshold ≥ 0.80.
task_medium : Low solar, high demand. Threshold ≥ 0.60.
task_hard   : Zero solar, grid failure. Threshold ≥ 0.30.
"""

from tasks.task_easy import run_task as run_easy, grade as grade_easy
from tasks.task_medium import run_task as run_medium, grade as grade_medium
from tasks.task_hard import run_task as run_hard, grade as grade_hard

__all__ = [
    "run_easy", "grade_easy",
    "run_medium", "grade_medium",
    "run_hard", "grade_hard",
]
