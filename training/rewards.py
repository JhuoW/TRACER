
from __future__ import annotations

import re


def extract_last_num(text: str) -> float:
    text = re.sub(r"(\d),(\d)", r"\g<1>\g<2>", text)
    res = re.findall(r"(\d+(\.\d+)?)", text)
    if len(res) > 0:
        return float(res[-1][0])
    return 0.0


def check(key: str, truth: str, predict: str) -> bool:
    if key in ['cycle', 'connectivity', 'hamilton', 'substructure', 'bipartite']:
        if '###' in predict:
            if 'yes' in truth.lower() and 'yes' in predict.split('###')[-1].lower():
                return True
            elif 'no' in truth.lower() and 'no' in predict.split('###')[-1].lower():
                return True
            return False
        else:
            matches = re.findall(r'\b(yes|no)\b', predict, flags=re.IGNORECASE)
            if matches:
                last_match = matches[-1].lower()
                if last_match == 'yes' and 'yes' in truth.lower():
                    return True
                elif last_match == 'no' and 'no' in truth.lower():
                    return True
            return False

    elif key in ['flow', 'shortest', 'triplet']:
        t_num = extract_last_num(truth)
        p_num = extract_last_num(predict.split('###')[-1])
        return abs(t_num - p_num) < 1e-2

    elif key == 'topology':
        if '###' in predict:
            pre = predict.split('###')[-1].strip(' ')
            truth_part = truth.split('###')[-1].strip(' ')
            return truth_part in pre or pre in truth_part
        else:
            truth_parts = truth.split('###')[-1].split(',')
            for t in truth_parts:
                if t in predict or t.strip(' ') in predict:
                    return True
            return False

    return False


def truncate_at_first_answer(text: str) -> str:
    parts = text.split("###")
    if len(parts) >= 2:
        return "###".join(parts[:2])
    return text


def _normalize_task(task: str) -> str:
    return "triplet" if task == "triangle" else task


def compute_reward(task: str, ground_truth: str, prediction: str) -> float:
    prediction = truncate_at_first_answer(prediction)
    return 1.0 if check(_normalize_task(task), ground_truth, prediction) else 0.0


def compute_shaped_reward(task: str, ground_truth: str, prediction: str) -> float:
    prediction = truncate_at_first_answer(prediction)

    task = _normalize_task(task)

    if check(task, ground_truth, prediction):
        return 1.0

    has_separator = "###" in prediction

    if task in ("flow", "shortest", "triplet"):
        if has_separator:
            t_num = extract_last_num(ground_truth)
            p_num = extract_last_num(prediction.split("###")[-1])
            denom = max(abs(t_num), 1.0)
            closeness = max(0.0, 1.0 - abs(p_num - t_num) / denom)
            return max(0.1, closeness)
        return 0.0

    return 0.1 if has_separator else 0.0
