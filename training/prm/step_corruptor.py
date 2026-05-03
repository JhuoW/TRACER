
from __future__ import annotations

import random
import re


def corrupt_step(
    step_text: str,
    task: str,
    max_node_id: int = 50,
    rng: random.Random | None = None,
    graph_nodes: set[int] | None = None,
) -> tuple[str, str]:
    if rng is None:
        rng = random.Random()

    is_conclusion = bool(
        re.search(r"(?:^|[.!?\s])(yes|no)\b\.?\s*$", step_text.strip(), re.IGNORECASE)
    )

    strategies = []

    node_ids = re.findall(r"\bnode (\d+)\b", step_text)
    if node_ids:
        strategies.append("node_swap")

    if task in ("shortest", "flow", "triangle", "triplet"):
        numbers = re.findall(r"(?<![a-zA-Z])(\d+)(?:\.\d+)?(?![a-zA-Z])", step_text)
        if numbers:
            strategies.append("arithmetic")

    if re.search(r"neighbors?[:\s].*\[.*\d+.*\]", step_text, re.IGNORECASE) or \
       re.search(r"connected to (?:node \d+(?:,\s*)?)+", step_text):
        strategies.append("neighbor_swap")

    if is_conclusion:
        strategies.append("logic_flip")

    weight_matches = re.findall(r"weight (?:of |is |= )?(\d+)", step_text)
    if weight_matches:
        strategies.append("weight_perturb")

    if not strategies:
        return step_text, "none"

    strategy = rng.choice(strategies)

    if strategy == "node_swap":
        return _corrupt_node_swap(step_text, max_node_id, rng, graph_nodes)
    elif strategy == "arithmetic":
        return _corrupt_arithmetic(step_text, rng)
    elif strategy == "neighbor_swap":
        return _corrupt_neighbor_swap(step_text, max_node_id, rng, graph_nodes)
    elif strategy == "logic_flip":
        return _corrupt_logic_flip(step_text, rng)
    elif strategy == "weight_perturb":
        return _corrupt_weight(step_text, rng)

    return step_text, "none"


def _corrupt_node_swap(
    text: str, max_node_id: int, rng: random.Random,
    graph_nodes: set[int] | None = None,
) -> tuple[str, str]:
    matches = list(re.finditer(r"\bnode (\d+)\b", text))
    if not matches:
        return text, "none"

    match = rng.choice(matches)
    old_id = int(match.group(1))

    if graph_nodes is not None and graph_nodes:
        lo, hi = min(graph_nodes), max(graph_nodes)
        oob_candidates = [hi + 1, hi + 2, max(0, lo - 1)]
        oob_candidates = [c for c in oob_candidates if c != old_id]
        if oob_candidates:
            new_id = rng.choice(oob_candidates)
        else:
            new_id = old_id
    else:
        new_id = old_id
        for _ in range(20):
            new_id = rng.randint(0, max_node_id)
            if new_id != old_id:
                break
    if new_id == old_id:
        return text, "none"

    result = text[:match.start(1)] + str(new_id) + text[match.end(1):]
    return result, "node_swap"


def _corrupt_arithmetic(text: str, rng: random.Random) -> tuple[str, str]:
    matches = list(re.finditer(r"(?<=[\s=+\-*/])(\d+)(?=[,\s\]\)>}.]|$)", text))
    if not matches:
        matches = list(re.finditer(r"\b(\d+)\b", text))
    if not matches:
        return text, "none"

    match = rng.choice(matches)
    old_num = int(match.group(1))

    delta = rng.choice([-3, -2, -1, 1, 2, 3])
    new_num = max(0, old_num + delta)
    if new_num == old_num:
        new_num = old_num + 1

    result = text[:match.start(1)] + str(new_num) + text[match.end(1):]
    return result, "arithmetic"


def _corrupt_neighbor_swap(
    text: str, max_node_id: int, rng: random.Random,
    graph_nodes: set[int] | None = None,
) -> tuple[str, str]:
    span_match = re.search(
        r"(neighbors?[:\s][^\n.]*|connected to[^\n.]*)",
        text, re.IGNORECASE,
    )
    if not span_match:
        return text, "none"

    span_start, span_end = span_match.span(1)
    digit_matches = list(re.finditer(r"\d+", text[span_start:span_end]))
    if not digit_matches:
        return text, "none"

    d = rng.choice(digit_matches)
    old_id = int(d.group(0))

    if graph_nodes is not None and graph_nodes:
        lo, hi = min(graph_nodes), max(graph_nodes)
        oob = [hi + 1, hi + 2, max(0, lo - 1)]
        oob = [c for c in oob if c != old_id]
        new_id = rng.choice(oob) if oob else old_id
    else:
        new_id = old_id
        for _ in range(20):
            new_id = rng.randint(0, max_node_id)
            if new_id != old_id:
                break
    if new_id == old_id:
        return text, "none"

    abs_start = span_start + d.start()
    abs_end = span_start + d.end()
    result = text[:abs_start] + str(new_id) + text[abs_end:]
    return result, "neighbor_swap"


def _corrupt_logic_flip(text: str, rng: random.Random) -> tuple[str, str]:
    if re.search(r"\byes\b", text, re.IGNORECASE):
        result = re.sub(r"\bYes\b", "No", text)
        result = re.sub(r"\byes\b", "no", result)
        return result, "logic_flip"
    elif re.search(r"\bno\b", text, re.IGNORECASE):
        result = re.sub(r"\bNo\b", "Yes", text)
        result = re.sub(r"\bno\b", "yes", result)
        return result, "logic_flip"
    return text, "none"


def _corrupt_weight(text: str, rng: random.Random) -> tuple[str, str]:
    matches = list(re.finditer(r"weight (?:of |is |= )?(\d+)", text))
    if not matches:
        return text, "none"

    match = rng.choice(matches)
    old_val = int(match.group(1))
    delta = rng.choice([-2, -1, 1, 2])
    new_val = max(1, old_val + delta)

    result = text[:match.start(1)] + str(new_val) + text[match.end(1):]
    return result, "weight_perturb"