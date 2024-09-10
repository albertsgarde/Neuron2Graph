import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Callable

import numpy as np
from graphviz import Digraph, escape  # type: ignore
from jaxtyping import Float
from pydantic import BaseModel, ConfigDict, NonNegativeInt

from .neuron_store import NeuronStore

ROOT_TOKEN = "**ROOT**"
IGNORE_TOKEN = "**IGNORE**"
END_TOKEN = "**END**"

SPECIAL_TOKENS = {ROOT_TOKEN, IGNORE_TOKEN, END_TOKEN}


@dataclass
class Sample:
    importances: Float[np.ndarray, "prompt_length prompt_length"]
    tokens_and_activations: list[tuple[str, float]]

    def tuple(
        self,
    ) -> tuple[Float[np.ndarray, "prompt_length prompt_length"], list[tuple[str, float]]]:
        return self.importances, self.tokens_and_activations


class Element(BaseModel):
    model_config = ConfigDict(strict=True)

    importance: float
    activation: float
    token: str
    important: bool
    activator: bool
    ignore: bool
    is_end: bool

    def __post_init__(self):
        assert self.token == self.token_value


class NeuronNode:
    id_: int
    value: Element
    depth: int

    children: dict[str, "NeuronNode"]

    def __init__(
        self,
        id_: int,
        value: Element,
        depth: int,
    ):
        self.children: dict[str, NeuronNode] = {}
        self.id_ = id_
        self.value = value
        self.depth = depth

    def __repr__(self):
        return f"ID: {self.id_}, Value: {json.dumps(self.value)}"

    def paths(self) -> list[list[Element]]:
        if not self.children:  # If the node has no children
            return [[self.value]]  # one path: only contains self.value
        paths: list[list[Element]] = []
        for _, child_node in self.children.items():
            for path in child_node.paths():
                paths.append([self.value] + path)
        return paths


def important_index_sets(
    sample: Sample,
    importance_threshold: float,
    ignore_end_of_text: bool,
) -> list[set[int]]:
    importances_matrix, tokens_and_activations = sample.tuple()
    result: list[set[int]] = []

    end_of_text_indices = np.array(
        [ignore_end_of_text and token == "<|endoftext|>" for token, _ in tokens_and_activations]
    )
    important_indices = np.tril((importances_matrix > importance_threshold).T & ~end_of_text_indices)

    result = [set(np.where(row)[0]) for row in important_indices]

    return result


Line = list[Element]


def make_lines(
    sample: Sample,
    important_index_sets: list[set[int]],
    importance_threshold: float,
    activation_threshold: float,
    ignore_end_of_text: bool,
) -> list[Line]:
    """
    Creates a list of patterns to be added to the neuron model.
    """
    importances_matrix, tokens_and_activations = sample.tuple()

    all_lines: list[Line] = []

    for i, (_, activation) in enumerate(tokens_and_activations):
        if not activation > activation_threshold:
            continue

        before = tokens_and_activations[: i + 1]

        line: Line = []
        last_important = 0

        # The if else is a bit of a hack to account for augmentations that have a different number
        # of tokens to the original prompt
        important_indices = important_index_sets[i] if i < len(important_index_sets) else important_index_sets[-1]

        for j, (seq_token, seq_activation) in enumerate(reversed(before)):
            if ignore_end_of_text and seq_token == "<|endoftext|>":
                continue

            seq_index = len(before) - j - 1
            importance = importances_matrix[seq_index, i]
            importance = float(importance)

            important = importance > importance_threshold or (seq_index in important_indices)
            activator = seq_activation > activation_threshold

            ignore = not important and j != 0
            is_end = False

            seq_token_identifier = IGNORE_TOKEN if ignore else seq_token

            new_element = Element(
                importance=importance,
                activation=seq_activation,
                token=seq_token_identifier,
                important=important,
                activator=activator,
                ignore=ignore,
                is_end=is_end,
            )

            if not ignore:
                last_important = j

            line.append(new_element)

        line = line[: last_important + 1]
        # Add an end node
        line.append(
            Element(
                importance=0,
                activation=activation,
                token=END_TOKEN,
                important=False,
                activator=False,
                ignore=True,
                is_end=True,
            )
        )
        all_lines.append(line)

    return all_lines


def samples_to_lines(
    samples: list[list[Sample]],
    importance_threshold: float,
    activation_threshold: float,
    ignore_end_of_text: bool,
) -> list[Line]:
    lines = []
    for sample_set in samples:
        original_sample = sample_set[0]
        original_sample_important_index_sets = important_index_sets(
            original_sample, importance_threshold, ignore_end_of_text
        )
        for sample in sample_set:
            lines += make_lines(
                sample,
                original_sample_important_index_sets,
                importance_threshold,
                activation_threshold,
                ignore_end_of_text,
            )
    return lines


class NeuronModel:
    root: NeuronNode
    trie_root: NeuronNode

    activation_threshold: float
    importance_threshold: float

    node_count: NonNegativeInt
    trie_node_count: NonNegativeInt
    max_depth: NonNegativeInt

    def __init__(
        self,
        activation_threshold: float,
        importance_threshold: float,
    ):
        self.root = NeuronNode(
            -1,
            Element(
                importance=0,
                activation=0,
                token=ROOT_TOKEN,
                important=False,
                activator=False,
                ignore=True,
                is_end=False,
            ),
            -1,
        )

        self.trie_root = NeuronNode(
            -1,
            Element(
                importance=0,
                activation=0,
                token=ROOT_TOKEN,
                important=False,
                activator=False,
                ignore=True,
                is_end=False,
            ),
            -1,
        )
        self.activation_threshold = activation_threshold
        self.importance_threshold = importance_threshold
        self.node_count = 0
        self.trie_node_count = 0
        self.max_depth: int = 0

    def __call__(self, tokens_arr: list[list[str]]) -> list[list[float]]:
        return self.forward(tokens_arr)

    @staticmethod
    def _normalise(token: str) -> str:
        return token

    def _add(
        self,
        start_node: NeuronNode,
        line: list[Element],
        graph: bool = True,
    ) -> None:
        current_node = start_node
        important_count = 0

        start_depth = current_node.depth

        for i, element in enumerate(line):
            if element.ignore and graph:
                continue

            important_count += 1

            # Normalise token
            element.token = NeuronModel._normalise(element.token)

            if graph:
                # Set end value as we don't have end nodes in the graph
                # The current node is an end if there's only one more node,
                # as that will be the end node that we don't add
                is_end = i == len(line) - 2
                element.is_end = is_end

            if element.token in current_node.children:
                current_node = current_node.children[element.token]
                continue

            depth = start_depth + important_count
            new_node = NeuronNode(self.node_count, element, depth)

            self.max_depth = depth if depth > self.max_depth else self.max_depth

            current_node.children[element.token] = new_node

            current_node = new_node

            self.node_count += 1

    def _merge_ignores(self) -> None:
        """
        Where a set of children contain an ignore token, merge the other nodes into it:
          - Fully merge if the other node is not an end node
          - Give the ignore node the other node's children (if it has any) if the other node is an end node
        """
        visited: set[int] = set()  # list to keep track of visited nodes.
        queue: list[NeuronNode] = []  # Initialize a queue

        visited.add(self.trie_root.id_)
        queue.append(self.trie_root)

        while queue:
            node = queue.pop(0)

            if IGNORE_TOKEN in node.children:
                ignore_tuple = node.children[IGNORE_TOKEN]

                to_remove: list[str] = []

                for child_token, child_node in node.children.items():
                    if child_token == IGNORE_TOKEN:
                        continue

                    child_paths = child_node.paths()

                    for path in child_paths:
                        # Don't merge if the path is only the first tuple, or the first tuple and an end tuple
                        if len(path) <= 1 or (len(path) == 2 and path[-1].token == END_TOKEN):
                            continue
                        # Merge the path (not including the first tuple that we're merging)
                        self._add(ignore_tuple, path[1:], graph=False)

                    # Add the node to a list to be removed later if it isn't an end node
                    # and doesn't have an end node in its children
                    if not child_node.value.is_end and END_TOKEN not in child_node.children:
                        to_remove.append(child_token)

                for child_token in to_remove:
                    node.children.pop(child_token)

            for _token, new_node in node.children.items():
                assert new_node.id_ not in visited, "Temporary assertion to check if hypothesis holds"
                if new_node.id_ not in visited:
                    visited.add(new_node.id_)
                    queue.append(new_node)

    def fit(
        self,
        samples: list[list[Sample]],
        ignore_end_of_text: bool,
    ):
        lines = samples_to_lines(samples, self.importance_threshold, self.activation_threshold, ignore_end_of_text)

        for line in lines:
            self._add(self.root, line, graph=True)
            self._add(self.trie_root, line, graph=False)
        self._merge_ignores()

    def _search(self, tokens: list[str]) -> float:
        """Evaluate the activation on the first token in tokens"""
        current_tuple = self.trie_root

        activations = [0.0]

        for i, token in enumerate(reversed(tokens)):
            token = NeuronModel._normalise(token)

            current_node = current_tuple

            if token in current_node.children or IGNORE_TOKEN in current_node.children:
                current_tuple = (
                    current_node.children[token]
                    if token in current_node.children
                    else current_node.children[IGNORE_TOKEN]
                )

                node = current_tuple
                # If the first token is not an activator, return early
                if i == 0:
                    if not node.value.activator:
                        break

                if END_TOKEN in node.children:
                    end_node = node.children[END_TOKEN]
                    end_activation = end_node.value.activation
                    activations.append(end_activation)

            else:
                break

        # Return the activation on the longest sequence
        return activations[-1]

    def forward(self, tokens_arr: list[list[str]]) -> list[list[float]]:
        if isinstance(tokens_arr[0], str):
            raise ValueError("tokens_arr must be of type list[list[str]]")

        """Evaluate the activation on each token in some input tokens"""
        all_activations: list[list[float]] = []

        for tokens in tokens_arr:
            activations: list[float] = []

            for j in range(len(tokens)):
                token_activation = self._search(tokens[: len(tokens) - j])
                activations.append(token_activation)

            activations = list(reversed(activations))

            all_activations.append(activations)

        return all_activations

    @staticmethod
    def _clamp(arr: tuple[int, int, int]):
        return [max(0, min(x, 255)) for x in arr]

    def graphviz(
        self,
    ) -> Digraph:
        """Build a graph to visualise"""
        visited: set[int] = set()  # list to keep track of visited nodes.
        # Second node is the first node's parent.
        queue: list[tuple[NeuronNode, NeuronNode | None]] = []  # Initialize a queue

        visited.add(self.root.id_)
        queue.append((self.root, None))

        zero_width = "\u200b"

        tokens_by_layer: dict[int, dict[str, str]] = {}
        node_id_to_graph_id: dict[int, str] = {}
        token_by_layer_count: dict[int, Counter[str]] = defaultdict(Counter)
        added_ids: set[str] = set()
        node_count: int = 0
        depth_to_subgraph: dict[int, Digraph] = {}
        added_edges: set[tuple[str, str]] = set()

        node_parent_tuples: list[tuple[NeuronNode, NeuronNode | None]] = []

        adjust: Callable[[float, float], float] = lambda x, y: (x - y) / (1 - y)

        while queue:
            (node, parent) = queue.pop(0)

            node_parent_tuples.append((node, parent))

            for _token, neighbour in node.children.items():
                new_node = neighbour
                if new_node.id_ not in visited:
                    visited.add(new_node.id_)
                    queue.append((neighbour, node))

        net = Digraph(
            graph_attr={
                "rankdir": "RL",
                "splines": "spline",
                "ranksep": "1.5",
                "nodesep": "0.2",
            },
            node_attr={"fixedsize": "true", "width": "2", "height": "0.75"},
        )

        for node, parent in node_parent_tuples:
            token = node.value.token
            depth = node.depth

            if depth not in tokens_by_layer:
                tokens_by_layer[depth] = {}
                depth_to_subgraph[depth] = Digraph(name=f"cluster_{str(self.max_depth - depth)}")
                depth_to_subgraph[depth].attr(pencolor="white", penwidth="3")  # type: ignore

            token_by_layer_count[depth][token] += 1

            if token not in tokens_by_layer[depth]:
                tokens_by_layer[depth][token] = str(node_count)
                node_count += 1

            graph_node_id = tokens_by_layer[depth][token]
            node_id_to_graph_id[node.id_] = graph_node_id

            current_graph = depth_to_subgraph[depth]

            if depth == 0:
                # colour red according to activation for depth 0 tokens
                scaled_activation = int(adjust(node.value.activation, max(0, self.activation_threshold - 0.2)) * 255)
                rgb = (255, 255 - scaled_activation, 255 - scaled_activation)
            else:
                # colour blue according to importance for all other tokens
                # Shift and scale importance so the importance threshold becomes 0

                scaled_importance = int(adjust(node.value.importance, max(0.1, self.importance_threshold - 0.2)) * 255)
                rgb = (255 - scaled_importance, 255 - scaled_importance, 255)

            hex = "#{0:02x}{1:02x}{2:02x}".format(*NeuronModel._clamp(rgb))

            if graph_node_id not in added_ids and not node.value.ignore:
                display_token = token.strip(zero_width)
                display_token = json.dumps(display_token).strip('[]"') if '"' not in token else display_token
                if set(display_token) == {" "}:
                    display_token = f"'{display_token}'"

                fontcolor = "white" if depth != 0 and rgb[1] < 130 else "black"
                fontsize = "25" if len(display_token) < 12 else "18"
                edge_width = "7" if node.value.is_end else "3"

                current_graph.node(  # type: ignore
                    graph_node_id,
                    f"{escape(display_token)}",
                    fillcolor=hex,
                    shape="box",
                    style="filled,solid",
                    fontcolor=fontcolor,
                    fontsize=fontsize,
                    penwidth=edge_width,
                )
                added_ids.add(graph_node_id)

            if parent is not None and parent.id_ in visited and not parent.value.ignore:
                graph_parent_id = node_id_to_graph_id[parent.id_]
                edge_tuple = (graph_parent_id, graph_node_id)
                if edge_tuple not in added_edges:
                    net.edge(*edge_tuple, penwidth="3", dir="back")  # type: ignore
                    added_edges.add(edge_tuple)

        for _depth, subgraph in depth_to_subgraph.items():
            net.subgraph(subgraph)  # type: ignore

        return net

    def update_neuron_store(self, neuron_store: NeuronStore, layer_name: str, neuron_index: int) -> None:
        visited: set[int] = set()  # list to keep track of visited nodes.
        queue: list[NeuronNode] = []  # Initialize a queue

        visited.add(self.trie_root.id_)
        queue.append(self.trie_root)

        while queue:
            node = queue.pop(0)

            token = node.value.token

            if token not in SPECIAL_TOKENS:
                neuron_store.add_neuron(node.value.activator, token, f"{layer_name}_{neuron_index}")

            for _token, neighbour in node.children.items():
                new_node = neighbour
                if new_node.id_ not in visited:
                    visited.add(new_node.id_)
                    queue.append(neighbour)
