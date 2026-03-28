"""
inference.py — LLM-driven agent for Project Blackout: Microgrid Power Dispatcher.

Environment variables (read from shell / .env):
    API_BASE_URL   Base URL for an OpenAI-compatible inference endpoint.
    MODEL_NAME     Model identifier to pass in the request.
    HF_TOKEN       Bearer token (used as the API key).

Usage:
    export API_BASE_URL="https://your-endpoint/v1"
    export MODEL_NAME="meta-llama/Llama-3-8b-instruct"
    export HF_TOKEN="hf_..."
    python inference.py --scenario medium
"""

import json
import logging
import os
import argparse
from typing import Optional

from openai import OpenAI

from env import PowerGridEnv
from grader import grade
from models import Action, DispatchType, Observation

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

MAX_STEPS: int = 10

# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

PROMPT_TEMPLATE = """\
You are a power grid controller.

Current State:
- Solar: {solar_mw} MW
- Battery SoC: {battery_soc}
- Demand: {demand_total} MW
- Grid Status: {grid_status}

Your goal is to keep the hospital (critical load) powered at all times.
You may charge the battery when solar is abundant, discharge it to cover demand gaps,
shed residential load as a last resort, or take no action (NOOP).

Return ONLY a valid JSON action — no extra text, no markdown:
{{
  "dispatch_type": "CHARGE" | "DISCHARGE" | "SHED_RESIDENTIAL" | "NOOP",
  "amount_mw": <float>
}}
"""


def build_prompt(obs: Observation) -> str:
    return PROMPT_TEMPLATE.format(
        solar_mw=obs.solar_mw,
        battery_soc=obs.battery_soc,
        demand_total=obs.demand_total,
        grid_status=obs.grid_status if isinstance(obs.grid_status, str) else obs.grid_status.value,
    )


# ---------------------------------------------------------------------------
# LLM client setup
# ---------------------------------------------------------------------------

def get_client() -> OpenAI:
    api_base = os.environ.get("API_BASE_URL", "").strip()
    hf_token = os.environ.get("HF_TOKEN", "").strip()

    if not api_base:
        raise EnvironmentError("API_BASE_URL environment variable is not set.")
    if not hf_token:
        raise EnvironmentError("HF_TOKEN environment variable is not set.")

    return OpenAI(base_url=api_base, api_key=hf_token)


def get_model_name() -> str:
    model = os.environ.get("MODEL_NAME", "").strip()
    if not model:
        raise EnvironmentError("MODEL_NAME environment variable is not set.")
    return model


# ---------------------------------------------------------------------------
# LLM action parsing
# ---------------------------------------------------------------------------

def query_llm(client: OpenAI, model: str, prompt: str) -> Optional[Action]:
    """
    Send a prompt to the LLM and parse the JSON response into an Action.
    Returns None (→ NOOP) on any failure.
    """
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=256,
        )
        raw_text: str = response.choices[0].message.content.strip()
        logger.debug("LLM raw response: %s", raw_text)
    except Exception as exc:
        logger.error("LLM request failed: %s", exc)
        return None

    # Strip possible markdown fences
    if raw_text.startswith("```"):
        lines = raw_text.splitlines()
        raw_text = "\n".join(
            line for line in lines if not line.strip().startswith("```")
        )

    try:
        payload = json.loads(raw_text)
        action = Action(
            dispatch_type=payload.get("dispatch_type", DispatchType.NOOP),
            amount_mw=float(payload.get("amount_mw", 0.0)),
        )
        logger.info("Parsed action: %s", action)
        return action
    except Exception as exc:
        logger.warning("Failed to parse LLM JSON (%s) — defaulting to NOOP. Raw: %s", exc, raw_text)
        return None


def safe_noop() -> Action:
    return Action(dispatch_type=DispatchType.NOOP, amount_mw=0.0)


# ---------------------------------------------------------------------------
# Main inference loop
# ---------------------------------------------------------------------------

def run_inference(scenario: str = "medium") -> None:
    client = get_client()
    model = get_model_name()

    env = PowerGridEnv()
    obs = env.reset(scenario=scenario)
    logger.info("Episode started | scenario=%s | initial obs=%s", scenario, obs)

    total_reward: float = 0.0

    for step in range(1, MAX_STEPS + 1):
        prompt = build_prompt(obs)
        action = query_llm(client, model, prompt) or safe_noop()

        result = env.step(action)
        total_reward += result.reward.score

        logger.info(
            "Step %d/%d | action=%s %.2f MW | reward=%.4f | hospital=%s | done=%s",
            step, MAX_STEPS,
            action.dispatch_type, action.amount_mw,
            result.reward.score,
            result.reward.is_hospital_powered,
            result.done,
        )

        obs = result.observation

        if result.done:
            logger.warning("Episode ended early at step %d (hospital blackout).", step)
            break

    # Grade the episode
    final_state = env.state()
    score = grade(final_state)

    print("\n" + "=" * 50)
    print(f"  Episode complete — {step} steps")
    print(f"  Total reward    : {total_reward:.4f}")
    print(f"  Grader score    : {score:.4f}  (1.0 = perfect)")
    print(f"  CO₂ saved       : {final_state['total_co2_saved']} tCO₂ proxy")
    print("=" * 50 + "\n")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run LLM agent on PowerGridEnv.")
    parser.add_argument(
        "--scenario",
        choices=["easy", "medium", "hard"],
        default="medium",
        help="Episode difficulty scenario (default: medium)",
    )
    args = parser.parse_args()
    run_inference(scenario=args.scenario)
