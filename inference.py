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
import sys
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

def get_llm_config() -> tuple[Optional[OpenAI], Optional[str]]:
    api_base = os.environ.get("API_BASE_URL", "").strip()
    hf_token = os.environ.get("HF_TOKEN", "").strip()
    model = os.environ.get("MODEL_NAME", "").strip()

    missing = [name for name, value in [
        ("API_BASE_URL", api_base),
        ("HF_TOKEN", hf_token),
        ("MODEL_NAME", model),
    ] if not value]

    if missing:
        logger.warning(
            "Missing environment variables (%s). Running fallback rule-based agent instead of LLM inference.",
            ", ".join(missing),
        )
        return None, None

    return OpenAI(base_url=api_base, api_key=hf_token), model


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


def heuristic_agent(obs: Observation) -> Action:
    """A simple fallback agent when LLM configuration is unavailable."""
    if obs.solar_mw >= obs.demand_total:
        return safe_noop()

    discharge_amount = min(5.0, max(0.0, obs.demand_total - obs.solar_mw))
    if obs.battery_soc > 0.20 and discharge_amount > 0.0:
        return Action(dispatch_type=DispatchType.DISCHARGE, amount_mw=discharge_amount)

    return Action(dispatch_type=DispatchType.SHED_RESIDENTIAL, amount_mw=discharge_amount)


def safe_noop() -> Action:
    return Action(dispatch_type=DispatchType.NOOP, amount_mw=0.0)


# ---------------------------------------------------------------------------
# Main inference loop
# ---------------------------------------------------------------------------

def run_inference(scenario: str = "medium") -> None:
    client, model = get_llm_config()

    env = PowerGridEnv()
    obs = env.reset(scenario=scenario)
    logger.info("Episode started | scenario=%s | initial obs=%s", scenario, obs)

    print(f"[START] task=inference scenario={scenario}", flush=True)

    total_reward: float = 0.0

    use_llm = client is not None and model is not None

    for step in range(1, MAX_STEPS + 1):
        prompt = build_prompt(obs)
        if use_llm:
            action = query_llm(client, model, prompt) or safe_noop()
        else:
            action = heuristic_agent(obs)

        result = env.step(action)
        total_reward += result.reward.score

        print(
            f"[STEP] step={step} action={action.dispatch_type} amount={action.amount_mw:.2f} "
            f"reward={result.reward.score:.4f} hospital={result.reward.is_hospital_powered} done={result.done}",
            flush=True,
        )

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

    print(
        f"[END] task=inference scenario={scenario} score={score:.4f} total_reward={total_reward:.4f} "
        f"steps={step} co2_saved={final_state['total_co2_saved']:.4f}",
        flush=True,
    )

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

    try:
        run_inference(scenario=args.scenario)
    except Exception as exc:
        logger.exception("Inference failed with an unexpected error.")
        print(
            "ERROR: Inference failed. Please verify that API_BASE_URL, HF_TOKEN, and MODEL_NAME are set if you want LLM inference.",
            file=sys.stderr,
        )
        sys.exit(1)
