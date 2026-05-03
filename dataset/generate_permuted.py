#!/usr/bin/env python3

import json
import re
import random
import argparse
from pathlib import Path


GRAPH_REPR = "[GRAPH_REPR]"

DIRECTED_TASKS = frozenset({"bipartite", "flow", "topology", "substructure"})
WEIGHTED_TASKS = frozenset({"shortest", "flow", "triplet", "triangle", "diameter"})
NODE_LIST_ANSWER_TASKS = frozenset({"topology", "hamilton"})

Q_ANCHORS = [
    r"Is there a path between",
    r"Is there a cycle",
    r"Does the graph contain a cycle",
    r"Give the weight of the shortest path",
    r"Is this graph bipartite",
    r"What is the diameter",
    r"What is the maximum flow",
    r"Is there a Hamiltonian path",
    r"What is the maximum sum",
    r"Give one topology sorting path",
    r"Is subgraph G",
]

_PH = "\x00{}\x00"
_LPH = "\x01{}\x01"


def node_count(query: str) -> int | None:
    m = re.search(r"nodes (?:are|of graph G are) numbered from (\d+) to (\d+)", query)
    return int(m.group(2)) + 1 if m else None


def subgraph_letters(query: str) -> list[str] | None:
    m = re.search(r"subgraph G' are numbered from ([a-z]) to ([a-z])", query)
    if m:
        return [chr(c) for c in range(ord(m.group(1)), ord(m.group(2)) + 1)]
    return None


def rand_perm(n: int) -> list[int]:
    p = list(range(n))
    random.shuffle(p)
    return p


def rand_letter_perm(letters: list[str]) -> dict[str, str]:
    s = letters[:]
    random.shuffle(s)
    return dict(zip(letters, s))


def _ph(perm, nid, n):
    return _PH.format(perm[nid]) if 0 <= nid < n else str(nid)


def _permute_numeric_edges(edge_str: str, perm: list[int],
                           directed: bool, weighted: bool) -> str:
    raw = re.findall(r"\([^)]+\)", edge_str)
    out: list[str] = []

    for t in raw:
        inner = t[1:-1]

        if directed and weighted:
            m = re.match(r"(\d+)\s*->\s*(\d+)\s*,\s*(\S+)", inner)
            if m:
                out.append(f"({perm[int(m.group(1))]}->{perm[int(m.group(2))]},{m.group(3)})")
                continue

        if directed:
            m = re.match(r"(\d+)\s*->\s*(\d+)", inner)
            if m:
                out.append(f"({perm[int(m.group(1))]}->{perm[int(m.group(2))]})")
                continue

        if weighted:
            m = re.match(r"(\d+)\s*,\s*(\d+)\s*,\s*(\S+)", inner)
            if m:
                out.append(f"({perm[int(m.group(1))]},{perm[int(m.group(2))]},{m.group(3)})")
                continue

        m = re.match(r"(\d+)\s*,\s*(\d+)", inner)
        if m:
            out.append(f"({perm[int(m.group(1))]},{perm[int(m.group(2))]})")
            continue

        out.append(t)

    random.shuffle(out)
    return " ".join(out)


def _permute_letter_edges(edge_str: str, lp: dict[str, str]) -> str:
    raw = re.findall(r"\([^)]+\)", edge_str)
    out: list[str] = []

    for t in raw:
        inner = t[1:-1]
        m = re.match(r"([a-z])\s*->\s*([a-z])", inner)
        if m:
            out.append(f"({lp[m.group(1)]}->{lp[m.group(2)]})")
            continue
        m = re.match(r"([a-z])\s*,\s*([a-z])", inner)
        if m:
            out.append(f"({lp[m.group(1)]},{lp[m.group(2)]})")
            continue
        out.append(t)

    random.shuffle(out)
    return " ".join(out)


def _insert_repr(query: str) -> str:
    q_pos = query.find("Q:")
    search_in = query[q_pos:] if q_pos >= 0 else query
    offset = q_pos if q_pos >= 0 else 0

    for pat in Q_ANCHORS:
        m = re.search(pat, search_in)
        if m:
            abs_pos = offset + m.start()
            before = query[:abs_pos].rstrip()
            after = query[abs_pos:]
            if not before.endswith("."):
                before += "."
            return f"{before} {GRAPH_REPR} {after}"
    return query


def transform_query(query: str, task: str, perm: list[int], n: int,
                    letter_perm: dict[str, str] | None = None,
                    insert_repr: bool = True) -> str:
    directed = task in DIRECTED_TASKS
    weighted = task in WEIGHTED_TASKS
    q = query


    if task == "substructure":
        m1 = re.search(
            r"(the edges are:\s*)(.*?)(\.\s*\n?\s*The nodes of subgraph)",
            q, re.DOTALL,
        )
        if m1:
            new_e = _permute_numeric_edges(m1.group(2), perm, True, False)
            q = q[: m1.start(2)] + new_e + q[m1.end(2) :]

        if letter_perm:
            m_range = re.search(
                r"(subgraph G' are numbered from )([a-z])( to )([a-z])", q
            )
            if m_range:
                new_labels = sorted(letter_perm.values())
                q = (
                    q[: m_range.start(2)]
                    + new_labels[0]
                    + q[m_range.end(2) : m_range.start(4)]
                    + new_labels[-1]
                    + q[m_range.end(4) :]
                )

            m2 = re.search(
                r"(the edges are:\s*)(.*?)(\.\s*Is subgraph)", q, re.DOTALL
            )
            if m2 and (not m1 or m2.start() > m1.start()):
                new_sub = _permute_letter_edges(m2.group(2), letter_perm)
                q = q[: m2.start(2)] + new_sub + q[m2.end(2) :]

    elif task in ("triplet", "triangle"):
        wm = re.search(
            r"(weights of nodes are:\s*)(.*?)(,\s*and the edges are)", q
        )
        if wm:
            entries = re.findall(r"\[(\d+),\s*(\d+)\]", wm.group(2))
            new_ents = sorted((perm[int(nid)], w) for nid, w in entries)
            new_w = " ".join(f"[{nid}, {w}]" for nid, w in new_ents)
            q = q[: wm.start(2)] + new_w + q[wm.end(2) :]

        em = re.search(r"(the edges are:\s*)(.*?)(\.\s*What)", q, re.DOTALL)
        if em:
            new_e = _permute_numeric_edges(em.group(2), perm, False, False)
            q = q[: em.start(2)] + new_e + q[em.end(2) :]

    else:
        em = re.search(
            r"(the edges are:\s*)(.*?)(\.\s*(?:Is |Give |What ))", q, re.DOTALL
        )
        if not em:
            em = re.search(r"(the edges are:\s*)(.*?)(\.\s)", q, re.DOTALL)
        if em:
            new_e = _permute_numeric_edges(em.group(2), perm, directed, weighted)
            q = q[: em.start(2)] + new_e + q[em.end(2) :]

    def _node_ref(m):
        nid = int(m.group(2))
        return m.group(1) + str(perm[nid]) if 0 <= nid < n else m.group(0)

    q = re.sub(r"([Nn]ode\s+)(\d+)", _node_ref, q)

    if insert_repr:
        q = _insert_repr(q)
    return q


def transform_answer(answer: str, task: str, perm: list[int], n: int,
                     letter_perm: dict[str, str] | None = None) -> str:
    if "###" not in answer:
        return _remap_trace(answer, perm, n, letter_perm)

    parts = answer.split("###")
    trace = "###".join(parts[:-1])
    final = parts[-1]

    new_trace = _remap_trace(trace, perm, n, letter_perm)
    new_final = (
        _remap_node_list(final, perm, n)
        if task in NODE_LIST_ANSWER_TASKS
        else final
    )
    return new_trace + "###" + new_final


def _remap_trace(text: str, perm: list[int], n: int,
                 lp: dict[str, str] | None = None) -> str:
    result = text

    def _nw(m):
        nid = int(m.group(2))
        return m.group(1) + _PH.format(perm[nid]) if 0 <= nid < n else m.group(0)

    result = re.sub(r"([Nn]ode\s+)(\d+)", _nw, result)

    def _edge_tuple(m):
        inner = m.group(0)[1:-1]

        dm = re.match(r"(\d+)\s*->\s*(\d+)\s*,\s*(\S+)", inner)
        if dm:
            u, v = int(dm.group(1)), int(dm.group(2))
            return f"({_ph(perm,u,n)}->{_ph(perm,v,n)},{dm.group(3)})"

        dm = re.match(r"(\d+)\s*->\s*(\d+)$", inner)
        if dm:
            u, v = int(dm.group(1)), int(dm.group(2))
            return f"({_ph(perm,u,n)}->{_ph(perm,v,n)})"

        dm = re.match(r"(\d+)\s*,\s*(\d+)\s*,\s*(\S+)$", inner)
        if dm:
            u, v = int(dm.group(1)), int(dm.group(2))
            return f"({_ph(perm,u,n)},{_ph(perm,v,n)},{dm.group(3)})"

        dm = re.match(r"(\d+)\s*,\s*(\d+)$", inner)
        if dm:
            u, v = int(dm.group(1)), int(dm.group(2))
            return f"({_ph(perm,u,n)},{_ph(perm,v,n)})"

        return m.group(0)

    result = re.sub(r"\([^)]+\)", _edge_tuple, result)

    def _bracket_list(m):
        content = m.group(1)
        def _rn(nm):
            nid = int(nm.group())
            return _PH.format(perm[nid]) if 0 <= nid < n else nm.group()
        return "[" + re.sub(r"\d+", _rn, content) + "]"

    result = re.sub(r"\[([\d,\s>-]+)\]", _bracket_list, result)

    if lp:
        for old, new in lp.items():
            result = result.replace(f"'{old}'", f"'{_LPH.format(new)}'")
        result = re.sub(r"\x01(.)\x01", r"\1", result)

    result = re.sub(r"\x00(\d+)\x00", r"\1", result)
    return result


def _remap_node_list(text: str, perm: list[int], n: int) -> str:
    def _bracket(m):
        content = m.group(1)
        def _rn(nm):
            nid = int(nm.group())
            return _PH.format(perm[nid]) if 0 <= nid < n else nm.group()
        return "[" + re.sub(r"\d+", _rn, content) + "]"

    result = re.sub(r"\[([\d,\s]+)\]", _bracket, text)
    result = re.sub(r"\x00(\d+)\x00", r"\1", result)
    return result


def augment_sample(sample: dict, k: int, insert_repr: bool = True) -> list[dict]:
    query = sample["query"]
    answer = sample["answer"]
    task = sample.get("task", "")
    if task == "triangle":
        task = "triplet"

    n = node_count(query)
    if n is None:
        q = _insert_repr(query) if insert_repr else query
        return [dict(sample, query=q, augmentation="original")]

    sub_ltrs = subgraph_letters(query) if task == "substructure" else None

    out: list[dict] = []

    orig = dict(sample)
    orig["query"] = _insert_repr(query) if insert_repr else query
    orig["augmentation"] = "original"
    out.append(orig)

    for i in range(k):
        perm = rand_perm(n)
        lp = rand_letter_perm(sub_ltrs) if sub_ltrs else None

        try:
            new_q = transform_query(query, task, perm, n, lp,
                                    insert_repr=insert_repr)
            new_a = transform_answer(answer, task, perm, n, lp)
        except Exception as exc:
            print(f"  [WARN] augmentation failed for index={sample.get('original_index','?')}, "
                  f"task={task}, perm #{i}: {exc}")
            aug = dict(sample)
            aug["query"] = _insert_repr(query) if insert_repr else query
            aug["answer"] = answer
            aug["augmentation"] = f"perm_{i}_fallback"
            out.append(aug)
            continue

        aug = dict(sample)
        aug["query"] = new_q
        aug["answer"] = new_a
        aug["augmentation"] = f"perm_{i}"
        out.append(aug)

    return out


def _load_hf_dataset():
    from datasets import load_dataset
    print("Loading GraphInstruct from HuggingFace …")
    ds = load_dataset("GraphWiz/GraphInstruct", split="train")
    print(f"  {len(ds):,} samples loaded")
    return ds


def _load_hf_rft_dataset():
    from datasets import load_dataset
    print("Loading GraphInstruct-RFT-72K from HuggingFace …")
    ds = load_dataset("GraphWiz/GraphInstruct-RFT-72K", split="train")
    print(f"  {len(ds):,} samples loaded")
    return ds


def cmd_permuted(args):
    random.seed(args.seed)
    ds = _load_hf_dataset()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "train.jsonl"

    total = 0
    task_counts: dict[str, int] = {}

    with open(out_path, "w") as f:
        for idx, sample in enumerate(ds):
            s = dict(sample)
            s["original_index"] = idx

            for aug in augment_sample(s, args.k):
                f.write(json.dumps(aug, ensure_ascii=False) + "\n")
                total += 1
                t = aug.get("task", "unknown")
                task_counts[t] = task_counts.get(t, 0) + 1

            if (idx + 1) % 2000 == 0:
                print(f"  {idx + 1:>6,}/{len(ds):,}  ({total:,} augmented)")

    print(f"\nDone — {total:,} samples written to {out_path}")
    print(f"  (original {len(ds):,} × {1 + args.k} = {len(ds) * (1 + args.k):,} expected)")
    print("\nPer-task breakdown:")
    for t in sorted(task_counts):
        print(f"  {t:20s}  {task_counts[t]:>8,}")

    meta = {
        "source": "GraphWiz/GraphInstruct",
        "k": args.k,
        "seed": args.seed,
        "total_samples": total,
        "task_counts": task_counts,
        "graph_repr_token": GRAPH_REPR,
    }
    meta_path = out_dir / "metadata.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Metadata saved to {meta_path}")


def cmd_aug(args):
    random.seed(args.seed)
    ds = _load_hf_dataset()

    permuted_path = Path(args.permuted) / "train.jsonl"
    if not permuted_path.exists():
        print(f"Permuted dataset not found at {permuted_path}")
        print("Run the 'permuted' command first:")
        print(f"  python {__file__} permuted")
        return

    print(f"Loading permuted dataset from {permuted_path} …")
    permuted: list[dict] = []
    with open(permuted_path) as f:
        for line in f:
            permuted.append(json.loads(line))
    print(f"  {len(permuted):,} permuted samples loaded")

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "train.jsonl"

    total = 0
    task_counts: dict[str, int] = {}

    with open(out_path, "w") as f:
        for idx, sample in enumerate(ds):
            row = dict(sample)
            row["query"] = _insert_repr(row["query"])
            row["original_index"] = idx
            row["augmentation"] = "none"
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            total += 1
            t = row.get("task", "unknown")
            task_counts[t] = task_counts.get(t, 0) + 1

        for row in permuted:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            total += 1
            t = row.get("task", "unknown")
            task_counts[t] = task_counts.get(t, 0) + 1

    n_orig = len(ds)
    n_perm = len(permuted)
    print(f"\nDone — {total:,} samples written to {out_path}")
    print(f"  Original (with [GRAPH_REPR]): {n_orig:>8,}")
    print(f"  Permuted:                     {n_perm:>8,}")
    print(f"  Total:                        {total:>8,}")
    print("\nPer-task breakdown:")
    for t in sorted(task_counts):
        print(f"  {t:20s}  {task_counts[t]:>8,}")

    meta = {
        "name": "GraphInstruct-Aug",
        "components": {
            "original": {
                "source": "GraphWiz/GraphInstruct",
                "samples": n_orig,
                "has_graph_repr": True,
            },
            "permuted": {
                "source": str(permuted_path),
                "samples": n_perm,
                "has_graph_repr": True,
            },
        },
        "total_samples": total,
        "task_counts": task_counts,
        "graph_repr_token": GRAPH_REPR,
    }
    meta_path = out_dir / "metadata.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Metadata saved to {meta_path}")


def cmd_rft(args):
    random.seed(args.seed)
    ds = _load_hf_rft_dataset()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "train.jsonl"

    total = 0
    task_counts: dict[str, int] = {}
    label_counts: dict[str, dict[str, int]] = {}

    with open(out_path, "w") as f:
        for idx, sample in enumerate(ds):
            s = dict(sample)
            s["original_index"] = idx

            for aug in augment_sample(s, args.k,
                                      insert_repr=not args.no_graph_repr):
                if aug["augmentation"] == "original":
                    aug["augmentation"] = "none"
                f.write(json.dumps(aug, ensure_ascii=False) + "\n")
                total += 1

                t = aug.get("task", "unknown")
                task_counts[t] = task_counts.get(t, 0) + 1

                if aug["augmentation"] == "none":
                    answer = aug.get("answer", "")
                    if "###" in answer:
                        final = answer.split("###")[-1].strip().lower()
                        label = "yes" if "yes" in final else ("no" if "no" in final else "other")
                    else:
                        label = "other"
                    label_counts.setdefault(t, {})
                    label_counts[t][label] = label_counts[t].get(label, 0) + 1

            if (idx + 1) % 5000 == 0:
                print(f"  {idx + 1:>6,}/{len(ds):,}  ({total:,} augmented)")

    print(f"\nDone — {total:,} samples written to {out_path}")
    print(f"  (RFT {len(ds):,} × {1 + args.k} = {len(ds) * (1 + args.k):,} expected)")
    print("\nPer-task breakdown:")
    for t in sorted(task_counts):
        print(f"  {t:20s}  {task_counts[t]:>8,}")
    print("\nLabel balance (anchor samples only):")
    for t in sorted(label_counts):
        lc = label_counts[t]
        parts = ", ".join(f"{k}={v}" for k, v in sorted(lc.items()))
        print(f"  {t:20s}  {parts}")

    meta = {
        "name": "GraphInstruct-RFT-Aug",
        "source": "GraphWiz/GraphInstruct-RFT-72K",
        "k": args.k,
        "seed": args.seed,
        "total_samples": total,
        "task_counts": task_counts,
        "label_balance": label_counts,
        "graph_repr_token": None if args.no_graph_repr else GRAPH_REPR,
    }
    meta_path = out_dir / "metadata.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Metadata saved to {meta_path}")


def main():
    ap = argparse.ArgumentParser(
        description="Generate augmented GraphInstruct datasets"
    )
    sub = ap.add_subparsers(dest="command")

    p_perm = sub.add_parser(
        "permuted",
        help="Generate GraphInstruct-Permuted (k permuted copies per sample)",
    )
    p_perm.add_argument("--k", type=int, default=4,
                        help="Number of permuted copies per sample (default: 4)")
    p_perm.add_argument("--seed", type=int, default=42)
    p_perm.add_argument("--output", type=str,
                        default="dataset/GraphInstruct-Permuted")

    p_aug = sub.add_parser(
        "aug",
        help="Generate GraphInstruct-Aug (original + permuted combined)",
    )
    p_aug.add_argument("--seed", type=int, default=42)
    p_aug.add_argument("--permuted", type=str,
                       default="dataset/GraphInstruct-Permuted",
                       help="Path to GraphInstruct-Permuted directory")
    p_aug.add_argument("--output", type=str,
                       default="dataset/GraphInstruct-Aug")

    p_rft = sub.add_parser(
        "rft",
        help="Generate GraphInstruct-RFT-Aug (RFT-72K + permutation augmentations)",
    )
    p_rft.add_argument("--k", type=int, default=2,
                       help="Permuted copies per sample (default: 2, lower than "
                            "base since RFT already provides reasoning diversity)")
    p_rft.add_argument("--seed", type=int, default=42)
    p_rft.add_argument("--no_graph_repr", action="store_true",
                       help="Do NOT insert [GRAPH_REPR] token (eliminates "
                            "train/eval mismatch; L_repr uses last prompt token)")
    p_rft.add_argument("--output", type=str,
                       default="dataset/GraphInstruct-RFT-Aug")

    args = ap.parse_args()

    if args.command == "permuted":
        cmd_permuted(args)
    elif args.command == "aug":
        cmd_aug(args)
    elif args.command == "rft":
        cmd_rft(args)
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
