
from __future__ import annotations

import re


def parse_trace_into_steps(answer: str, task: str) -> list[dict]:
    if "###" in answer:
        trace, final = answer.split("###", 1)
    else:
        trace = answer
        final = ""

    trace = trace.strip()
    if not trace:
        return []

    raw_steps = _split_into_raw_steps(trace, task)

    raw_steps = [s.strip() for s in raw_steps if s.strip()]
    if len(raw_steps) < 2:
        return []

    sep = "\n" if any("\n" in s for s in raw_steps) else " "
    steps = []
    for i, step_text in enumerate(raw_steps):
        prefix = sep.join(raw_steps[:i])
        steps.append({
            "text": step_text,
            "step_idx": i,
            "prefix": prefix,
        })

    return steps


def _split_into_raw_steps(trace: str, task: str) -> list[str]:
    bullet_lines = re.findall(r"^- .+", trace, re.MULTILINE)
    if len(bullet_lines) >= 3:
        return _split_bullet_steps(trace)

    path_lines = re.findall(r"^\d+(?:,\d+)+ with a total weight", trace, re.MULTILINE)
    if len(path_lines) >= 2:
        return _split_path_enum_steps(trace)

    from_lines = re.findall(r"(?:^|\n)From node \d+", trace)
    if len(from_lines) >= 3:
        return _split_from_node_steps(trace)

    return _split_sentence_steps(trace)


def _split_bullet_steps(trace: str) -> list[str]:
    steps = []
    current = []
    for line in trace.split("\n"):
        if line.strip().startswith("- ") and current:
            steps.append("\n".join(current))
            current = [line.strip()]
        else:
            current.append(line.strip())
    if current:
        steps.append("\n".join(current))
    return steps


def _split_path_enum_steps(trace: str) -> list[str]:
    lines = [l.strip() for l in trace.split("\n") if l.strip()]
    steps = []
    intro = []
    paths = []
    conclusion = []
    in_paths = False

    for line in lines:
        if re.match(r"\d+(?:,\d+)+ with a total weight", line):
            in_paths = True
            paths.append(line)
        elif in_paths:
            conclusion.append(line)
        else:
            intro.append(line)

    if intro:
        steps.append(" ".join(intro))
    chunk_size = max(1, len(paths) // 4) if len(paths) > 4 else 1
    for i in range(0, len(paths), chunk_size):
        chunk = paths[i:i + chunk_size]
        steps.append("\n".join(chunk))
    if conclusion:
        steps.append(" ".join(conclusion))
    return steps


def _split_from_node_steps(trace: str) -> list[str]:
    parts = re.split(r"(?=(?:^|\n)From node \d+)", trace)
    steps = [p.strip() for p in parts if p.strip()]

    return steps


def _split_sentence_steps(trace: str) -> list[str]:
    sentences = re.split(r"(?<=[.!?])\s+", trace)

    merged = []
    for s in sentences:
        s = s.strip()
        if not s:
            continue
        if merged and len(s) < 30:
            merged[-1] = merged[-1] + " " + s
        else:
            merged.append(s)

    return merged