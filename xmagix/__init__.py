"""xmagix: a fast 2D cross-embodiment imitation benchmark on MuJoCo Warp (x-magical successor)."""

from xmagix.embodiments import EMBODIMENTS
from xmagix.env import Env, EnvState, make, rollout
from xmagix.tasks import TASKS

__all__ = ["EMBODIMENTS", "TASKS", "Env", "EnvState", "make", "rollout", "expert_success", "valid_pairs"]


def expert_success(variant="default"):
    """Measured scripted-expert success per 'task/embodiment' (scripts/expert_check.py, 256 episodes)."""
    import json
    from pathlib import Path

    return json.loads((Path(__file__).parent / f"expert_success_{variant}.json").read_text())


def valid_pairs(threshold=0.9, variant="default"):
    """(task, embodiment) pairs whose scripted expert solves the task at least `threshold` of the
    time: the pairs the benchmark uses (an unsolvable demonstration is not a valid test)."""
    return [tuple(k.split("/")) for k, v in expert_success(variant).items() if v >= threshold]
