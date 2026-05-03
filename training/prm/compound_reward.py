
from __future__ import annotations

from training.rewards import compute_shaped_reward


def compute_compound_reward(
    task: str,
    ground_truth: str,
    prediction: str,
    query: str,
    prm_scorer=None,
    w_trace: float = 0.3,
) -> float:
    r_shaped = compute_shaped_reward(task, ground_truth, prediction)

    if prm_scorer is None or w_trace <= 0:
        return r_shaped

    r_trace = prm_scorer.score_trace(query, prediction, task)
    return (1.0 - w_trace) * r_shaped + w_trace * r_trace


def compute_compound_rewards_batch(
    tasks: list[str],
    ground_truths: list[str],
    predictions: list[str],
    queries: list[str],
    prm_scorer=None,
    w_trace: float = 0.3,
) -> list[float]:
    shaped_rewards = [
        compute_shaped_reward(t, gt, p)
        for t, gt, p in zip(tasks, ground_truths, predictions)
    ]

    if prm_scorer is None or w_trace <= 0:
        return shaped_rewards

    trace_rewards = prm_scorer.score_traces_batch(
        queries, predictions, tasks,
    )

    return [
        (1.0 - w_trace) * r_s + w_trace * r_t
        for r_s, r_t in zip(shaped_rewards, trace_rewards)
    ]