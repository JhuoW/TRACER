
import re
import torch
import networkx as nx

DIRECTED_TASKS = frozenset({"bipartite", "flow", "topology", "substructure"})
WEIGHTED_EDGE_TASKS = frozenset({"shortest", "flow", "diameter"})

_Q_ANCHORS = [
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


def _find_question_boundary(text: str, search_start: int = 0) -> int:
    repr_pos = text.find("[GRAPH_REPR]", search_start)
    if repr_pos != -1:
        return repr_pos

    region = text[search_start:]
    for pat in _Q_ANCHORS:
        m = re.search(pat, region)
        if m:
            return search_start + m.start()

    return len(text)


class GraphParser:


    def parse(self, text: str, task: str = "") -> dict:
        directed = task in DIRECTED_TASKS
        weighted = task in WEIGHTED_EDGE_TASKS

        nodes: set[int] = set()
        edges: list[tuple] = []

        m = re.search(
            r"nodes (?:are|of graph G are) numbered from (\d+) to (\d+)", text
        )
        if m:
            nodes.update(range(int(m.group(1)), int(m.group(2)) + 1))

        edge_start = text.find("the edges are:")
        if edge_start == -1:
            return self._empty(directed, weighted)

        q_boundary = _find_question_boundary(text, edge_start)
        edge_region = text[edge_start : q_boundary]

        if directed and weighted:
            pat = r"\((\d+)\s*->\s*(\d+)\s*,\s*(\d+)\)"
            for m in re.finditer(pat, edge_region):
                u, v, w = int(m.group(1)), int(m.group(2)), int(m.group(3))
                edges.append((u, v, w))
                nodes.update([u, v])
        elif directed:
            pat = r"\((\d+)\s*->\s*(\d+)\)"
            for m in re.finditer(pat, edge_region):
                u, v = int(m.group(1)), int(m.group(2))
                edges.append((u, v))
                nodes.update([u, v])
        elif weighted:
            pat = r"\((\d+)\s*,\s*(\d+)\s*,\s*(\d+)\)"
            for m in re.finditer(pat, edge_region):
                u, v, w = int(m.group(1)), int(m.group(2)), int(m.group(3))
                edges.append((u, v, w))
                nodes.update([u, v])
        else:
            pat = r"\((\d+)\s*,\s*(\d+)\)"
            for m in re.finditer(pat, edge_region):
                u, v = int(m.group(1)), int(m.group(2))
                edges.append((u, v))
                nodes.update([u, v])

        return {
            "nodes": sorted(nodes),
            "edges": edges,
            "directed": directed,
            "weighted": weighted,
        }


    def compute_shortest_paths(
        self, graph: dict, d_max: int = 1
    ) -> dict[tuple[int, int], int]:
        if not graph["nodes"]:
            return {}

        G = nx.DiGraph() if graph["directed"] else nx.Graph()
        for n in graph["nodes"]:
            G.add_node(n)
        for edge in graph["edges"]:
            G.add_edge(edge[0], edge[1])

        distances: dict[tuple[int, int], int] = {}
        for u in graph["nodes"]:
            lengths = dict(
                nx.single_source_shortest_path_length(G, u, cutoff=d_max)
            )
            for v in graph["nodes"]:
                distances[(u, v)] = lengths.get(v, d_max + 1)

        return distances


    def build_entity_map(
        self, text: str, tokenizer, graph: dict
    ) -> dict[int, int]:
        encoding = tokenizer(
            text, return_offsets_mapping=True, add_special_tokens=False
        )
        return self._entity_map_from_offsets(
            text, encoding["offset_mapping"], graph
        )

    def build_entity_map_from_offsets(
        self, text: str, offsets: list[tuple[int, int]], graph: dict
    ) -> dict[int, int]:
        return self._entity_map_from_offsets(text, offsets, graph)

    def _entity_map_from_offsets(
        self,
        text: str,
        offsets: list[tuple[int, int]],
        graph: dict,
    ) -> dict[int, int]:
        if not graph["nodes"]:
            return {}

        node_set = set(graph["nodes"])
        directed = graph["directed"]
        weighted = graph["weighted"]

        spans: list[tuple[int, int, int]] = []

        q_start = text.find("Q:")
        if q_start == -1:
            q_start = 0

        edge_marker = text.find("the edges are:", q_start)
        if edge_marker == -1:
            return {}

        q_boundary = _find_question_boundary(text, edge_marker)
        edge_end = q_boundary
        edge_region = text[edge_marker:edge_end]
        eo = edge_marker

        if directed and weighted:
            pat = r"\((\d+)\s*->\s*(\d+)\s*,\s*\d+\)"
            groups = [1, 2]
        elif directed:
            pat = r"\((\d+)\s*->\s*(\d+)\)"
            groups = [1, 2]
        elif weighted:
            pat = r"\((\d+)\s*,\s*(\d+)\s*,\s*\d+\)"
            groups = [1, 2]
        else:
            pat = r"\((\d+)\s*,\s*(\d+)\)"
            groups = [1, 2]

        for m in re.finditer(pat, edge_region):
            for g in groups:
                nid = int(m.group(g))
                if nid in node_set:
                    spans.append((eo + m.start(g), eo + m.end(g), nid))

        for m in re.finditer(r"\[(\d+)\s*,\s*\d+\]", text[q_start:edge_end]):
            nid = int(m.group(1))
            if nid in node_set:
                spans.append(
                    (q_start + m.start(1), q_start + m.end(1), nid)
                )

        question_start = edge_end
        for m in re.finditer(r"(?i)node\s+(\d+)", text[question_start:]):
            nid = int(m.group(1))
            if nid in node_set:
                spans.append(
                    (question_start + m.start(1), question_start + m.end(1), nid)
                )

        entity_map: dict[int, int] = {}
        for char_start, char_end, nid in spans:
            for tok_idx, (ts, te) in enumerate(offsets):
                if ts is None or te is None or te == 0:
                    continue
                if ts < char_end and te > char_start:
                    entity_map[tok_idx] = nid

        return entity_map


    def build_bias_matrix(
        self,
        entity_map: dict[int, int],
        shortest_paths: dict[tuple[int, int], int],
        seq_len: int,
        d_max: int = 1,
    ) -> torch.Tensor:
        B = torch.full((seq_len, seq_len), -1, dtype=torch.long)

        positions = list(entity_map.keys())
        for i in positions:
            u = entity_map[i]
            for j in positions:
                v = entity_map[j]
                if u == v:
                    B[i, j] = 0
                else:
                    B[i, j] = shortest_paths.get((u, v), d_max + 1)

        return B


    def select_landmarks(self, graph: dict, r: int = 8) -> list[int]:
        if not graph["nodes"]:
            return []

        G = nx.DiGraph() if graph["directed"] else nx.Graph()
        for n in graph["nodes"]:
            G.add_node(n)
        for edge in graph["edges"]:
            G.add_edge(edge[0], edge[1])

        by_degree = sorted(graph["nodes"], key=lambda n: (-G.degree(n), n))
        return by_degree[:r]

    def compute_landmark_fingerprints(
        self,
        graph: dict,
        shortest_paths: dict[tuple[int, int], int],
        r: int = 8,
        d_max_fp: int = 10,
    ) -> tuple[dict[int, list[float]], list[int]]:
        landmarks = self.select_landmarks(graph, r)
        if not landmarks:
            return {}, []

        while len(landmarks) < r:
            landmarks.append(landmarks[-1])

        fingerprints: dict[int, list[float]] = {}
        for v in graph["nodes"]:
            fp = []
            for lm in landmarks:
                d = shortest_paths.get((v, lm), d_max_fp)
                d = min(d, d_max_fp)
                fp.append(d / d_max_fp)
            fingerprints[v] = fp

        return fingerprints, landmarks


    @staticmethod
    def _empty(directed: bool, weighted: bool) -> dict:
        return {
            "nodes": [],
            "edges": [],
            "directed": directed,
            "weighted": weighted,
        }
