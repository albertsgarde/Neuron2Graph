from collections import Counter, defaultdict, namedtuple
import json
import os
from typing import Any, List, NamedTuple

from graphviz import Digraph, escape

from n2g.neuron_store import NeuronStore


class Element(NamedTuple):
    importance: float
    activation: float
    token: str
    important: bool
    activator: bool
    ignore: bool
    is_end: bool
    token_value: str


class NeuronNode:
    def __init__(
        self,
        id_: int,
        value: Element,
        depth: int,
        children=None,
        important: bool = False,
        activator: bool = False,
    ):
        if value is None:
            value = {}
        if children is None:
            children = {}
        self.id_ = id_
        self.value = value
        self.children = children
        self.depth = depth

    def __repr__(self):
        return f"ID: {self.id_}, Value: {json.dumps(self.value)}"

    def paths(self):
        if not self.children:
            return [[self.value]]  # one path: only contains self.value
        paths = []
        for child_token, child_tuple in self.children.items():
            child_node, _ = child_tuple
            for path in child_node.paths():
                paths.append([self.value] + path)
        return paths


class NeuronEdge:
    def __init__(self, weight=0, parent=None, child=None):
        self.weight = weight
        self.parent = parent
        self.child = child

    def __repr__(self):
        parent_str = json.dumps(self.parent.id_) if self.parent is not None else "None"
        child_str = json.dumps(self.child.id_) if self.child is not None else "None"
        return f"Weight: {self.weight:.3f}\nParent: {parent_str}\nChild: {child_str}"


class NeuronModel:
    def __init__(
        self,
        layer,
        neuron,
        neuron_store: NeuronStore,
        activation_threshold=0.1,
        importance_threshold=0.5,
        folder_name=None,
        **kwargs,
    ):
        self.layer = layer
        self.neuron = neuron
        self.neuron_store = neuron_store

        self.root_token = "**ROOT**"
        self.ignore_token = "**IGNORE**"
        self.end_token = "**END**"
        self.special_tokens = {self.root_token, self.ignore_token, self.end_token}

        self.root = (
            NeuronNode(
                -1,
                Element(
                    0, 0, self.root_token, False, False, True, False, self.root_token
                ),
                -1,
            ),
            NeuronEdge(),
        )
        self.trie_root = (
            NeuronNode(
                -1,
                Element(
                    0, 0, self.root_token, False, False, True, False, self.root_token
                ),
                -1,
            ),
            NeuronEdge(),
        )
        self.activation_threshold = activation_threshold
        self.importance_threshold = importance_threshold
        # self.net = Network(notebook=True)
        # self.net = Graph(graph_attr={"rankdir": "LR", "splines": "spline", "ranksep": "20", "nodesep": "1"}, node_attr={"fixedsize": "true", "width": "1.5"})
        # self.net = Graph(
        #     graph_attr={"rankdir": "RL", "splines": "spline", "ranksep": "5", "nodesep": "1"},
        #     node_attr={"fixedsize": "true", "width": "2"}
        # )
        # self.net = Graph(
        #     graph_attr={"rankdir": "RL", "splines": "spline", "ranksep": "2", "nodesep": "0.25"},
        #     node_attr={"fixedsize": "true", "width": "2", "height": "0.75"}
        # )
        self.net = Digraph(
            graph_attr={
                "rankdir": "RL",
                "splines": "spline",
                "ranksep": "1.5",
                "nodesep": "0.2",
            },
            node_attr={"fixedsize": "true", "width": "2", "height": "0.75"},
        )
        self.node_count = 0
        self.trie_node_count = 0
        self.max_depth = 0
        self.folder_name = folder_name

    def __call__(self, tokens_arr: List[List[str]]) -> List[List[float]]:
        return self.forward(tokens_arr)

    def fit(self, data, base_path: str, model_name: str):
        for example_data in data:
            for j, info in enumerate(example_data):
                if j == 0:
                    lines, important_index_sets = self.make_line(info)
                else:
                    lines, _ = self.make_line(info, important_index_sets)

                for line in lines:
                    self.add(self.root, line, graph=True)
                    self.add(self.trie_root, line, graph=False)

        self.build(self.root, base_path, model_name)
        self.merge_ignores()

        self.save_neurons()

        print("Paths after merge", flush=True)
        paths = []
        for path in self.trie_root[0].paths():
            paths.append(path)

        return paths

    def save_neurons(self):
        visited = set()  # List to keep track of visited nodes.
        queue = []  # Initialize a queue

        visited.add(self.trie_root[0].id_)
        queue.append(self.trie_root)

        while queue:
            node, edge = queue.pop(0)

            token = node.value.token

            if token not in self.special_tokens:
                add_dict = (
                    self.neuron_store.store["activating"]
                    if node.value.activator
                    else self.neuron_store.store["important"]
                )
                if token not in add_dict:
                    add_dict[token] = set()
                else:
                    add_dict[token] = set(add_dict[token])
                add_dict[token].add(f"{self.layer}_{self.neuron}")

            for token, neighbour in node.children.items():
                new_node, new_edge = neighbour
                if new_node.id_ not in visited:
                    visited.add(new_node.id_)
                    queue.append(neighbour)

    @staticmethod
    def normalise(token):
        normalised_token = (
            token.lower() if token.istitle() and len(token) > 1 else token
        )
        normalised_token = (
            normalised_token.strip()
            if len(normalised_token) > 1 and any(c.isalpha() for c in normalised_token)
            else normalised_token
        )
        return normalised_token

    def make_line(self, info, important_index_sets=None):
        if important_index_sets is None:
            important_index_sets = []
            create_indices = True
        else:
            create_indices = False

        importances_matrix, tokens_and_activations, max_index = info

        all_lines: List[List[Element]] = []

        for i, (token, activation) in enumerate(tokens_and_activations):
            if create_indices:
                important_index_sets.append(set())

            if not activation > self.activation_threshold:
                continue

            before = tokens_and_activations[: i + 1]

            line = []
            last_important = 0

            if not create_indices:
                # The if else is a bit of a hack to account for augmentations that have a different number of tokens to the original prompt
                important_indices = (
                    important_index_sets[i]
                    if i < len(important_index_sets)
                    else important_index_sets[-1]
                )
            else:
                important_indices = set()

            for j, (seq_token, seq_activation) in enumerate(reversed(before)):
                if seq_token == "<|endoftext|>":
                    continue

                seq_index = len(before) - j - 1
                important_token, importance = importances_matrix[seq_index, i]
                importance = float(importance)

                important = importance > self.importance_threshold or (
                    not create_indices and seq_index in important_indices
                )
                activator = seq_activation > self.activation_threshold

                if important and create_indices:
                    important_indices.add(seq_index)

                ignore = not important and j != 0
                is_end = False

                seq_token_identifier = self.ignore_token if ignore else seq_token

                new_element = Element(
                    importance,
                    seq_activation,
                    seq_token_identifier,
                    important,
                    activator,
                    ignore,
                    is_end,
                    seq_token,
                )

                if not ignore:
                    last_important = j

                line.append(new_element)

            line = line[: last_important + 1]
            # Add an end node
            line.append(
                Element(
                    0,
                    activation,
                    self.end_token,
                    False,
                    False,
                    True,
                    True,
                    self.end_token,
                )
            )
            all_lines.append(line)

            if create_indices:
                important_index_sets[i] = important_indices

        return all_lines, important_index_sets

    def add(self, start_tuple, line, graph=True):
        current_tuple = start_tuple
        previous_element = None
        important_count = 0

        start_depth = current_tuple[0].depth

        for i, element in enumerate(line):
            if element is None and i > 0:
                break

            if element.ignore and graph:
                continue

            # Normalise token
            element = element._replace(token=self.normalise(element.token))

            if graph:
                # Set end value as we don't have end nodes in the graph
                # The current node is an end if there's only one more node, as that will be the end node that we don't add
                is_end = i == len(line) - 2
                element = element._replace(is_end=is_end)

            important_count += 1

            current_node, current_edge = current_tuple

            if not current_node.value.ignore:
                prev_important_node = current_node

            if element.token in current_node.children:
                current_tuple = current_node.children[element.token]
                continue

            weight = 0

            depth = start_depth + important_count
            new_node = NeuronNode(self.node_count, element, depth, {})
            new_tuple = (new_node, NeuronEdge(weight, current_node, new_node))

            self.max_depth = depth if depth > self.max_depth else self.max_depth

            current_node.children[element.token] = new_tuple

            current_tuple = new_tuple

            self.node_count += 1

        return current_tuple

    def merge_ignores(self):
        """
        Where a set of children contain an ignore token, merge the other nodes into it:
          - Fully merge if the other node is not an end node
          - Give the ignore node the other node's children (if it has any) if the other node is an end node
        """
        visited = set()  # List to keep track of visited nodes.
        queue = []  # Initialize a queue

        visited.add(self.trie_root[0].id_)
        queue.append(self.trie_root)

        while queue:
            node, edge = queue.pop(0)

            token = node.value.token

            if self.ignore_token in node.children:
                ignore_tuple = node.children[self.ignore_token]

                to_remove = []

                for child_token, child_tuple in node.children.items():
                    if child_token == self.ignore_token:
                        continue

                    child_node, child_edge = child_tuple

                    child_paths = child_node.paths()

                    for path in child_paths:
                        # Don't merge if the path is only the first tuple, or the first tuple and an end tuple
                        if len(path) <= 1 or (
                            len(path) == 2 and path[-1].token == self.end_token
                        ):
                            continue
                        # Merge the path (not including the first tuple that we're merging)
                        self.add(ignore_tuple, path[1:], graph=False)

                    # Add the node to a list to be removed later if it isn't an end node and doesn't have an end node in its children
                    if (
                        not child_node.value.is_end
                        and not self.end_token in child_node.children
                    ):
                        to_remove.append(child_token)

                for child_token in to_remove:
                    node.children.pop(child_token)

            for token, neighbour in node.children.items():
                new_node, new_edge = neighbour
                if new_node.id_ not in visited:
                    visited.add(new_node.id_)
                    queue.append(neighbour)

    def search(self, tokens: List[str]) -> float:
        """Evaluate the activation on the first token in tokens"""
        current_tuple = self.trie_root

        activations = [0]

        for i, token in enumerate(reversed(tokens)):
            token = self.normalise(token)

            current_node, current_edge = current_tuple

            if (
                token in current_node.children
                or self.ignore_token in current_node.children
            ):
                current_tuple = (
                    current_node.children[token]
                    if token in current_node.children
                    else current_node.children[self.ignore_token]
                )

                node, edge = current_tuple
                # If the first token is not an activator, return early
                if i == 0:
                    if not node.value.activator:
                        break
                    activation = node.value.activation

                if self.end_token in node.children:
                    end_node, _ = node.children[self.end_token]
                    end_activation = end_node.value.activation
                    activations.append(end_activation)

            else:
                break

        # Return the activation on the longest sequence
        return activations[-1]

    def forward(
        self, tokens_arr: List[List[str]], return_activations=True
    ) -> List[List[float]]:
        if isinstance(tokens_arr[0], str):
            raise ValueError(f"tokens_arr must be of type List[List[str]]")

        """Evaluate the activation on each token in some input tokens"""
        all_activations = []
        all_firings = []

        for tokens in tokens_arr:
            activations = []
            firings = []

            for j in range(len(tokens)):
                token_activation = self.search(tokens[: len(tokens) - j])
                activations.append(token_activation)
                firings.append(token_activation > self.activation_threshold)

            activations = list(reversed(activations))
            firings = list(reversed(firings))

            all_activations.append(activations)
            all_firings.append(firings)

        if return_activations:
            return all_activations
        return all_firings

    def build(self, start_node, base_path: str, model_name: str, graph=True):
        """Build a graph to visualise"""
        visited = set()  # List to keep track of visited nodes.
        queue = []  # Initialize a queue

        visited.add(start_node[0].id_)
        queue.append(start_node)

        zero_width = "\u200b"

        tokens_by_layer = {}
        node_id_to_graph_id = {}
        token_by_layer_count = defaultdict(Counter)
        added_ids = set()
        node_count = 0
        depth_to_subgraph = {}
        added_edges = set()

        node_edge_tuples = []

        adjust = lambda x, y: (x - y) / (1 - y)

        while queue:
            node, edge = queue.pop(0)

            node_edge_tuples.append((node, edge))

            for token, neighbour in node.children.items():
                new_node, new_edge = neighbour
                if new_node.id_ not in visited:
                    visited.add(new_node.id_)
                    queue.append(neighbour)

        for node, edge in node_edge_tuples:
            token = node.value.token
            depth = node.depth

            if depth not in tokens_by_layer:
                tokens_by_layer[depth] = {}
                depth_to_subgraph[depth] = Digraph(
                    name=f"cluster_{str(self.max_depth - depth)}"
                )
                depth_to_subgraph[depth].attr(pencolor="white", penwidth="3")

            token_by_layer_count[depth][token] += 1

            if not graph:
                # This is a horrible hack to allow us to have a dict with the "same" token as multiple keys - by adding zero width spaces the tokens look the same but are actually different. This allows us to display a trie rather than a node-collapsed graph
                seen_count = token_by_layer_count[depth][token] - 1
                add = zero_width * seen_count
                token += add

            if token not in tokens_by_layer[depth]:
                tokens_by_layer[depth][token] = str(node_count)
                node_count += 1

            graph_node_id = tokens_by_layer[depth][token]
            node_id_to_graph_id[node.id_] = graph_node_id

            current_graph = depth_to_subgraph[depth]

            if depth == 0:
                # colour red according to activation for depth 0 tokens
                scaled_activation = int(
                    adjust(
                        node.value.activation, max(0, self.activation_threshold - 0.2)
                    )
                    * 255
                )
                rgb = (255, 255 - scaled_activation, 255 - scaled_activation)
            else:
                # colour blue according to importance for all other tokens
                # Shift and scale importance so the importance threshold becomes 0

                scaled_importance = int(
                    adjust(
                        node.value.importance, max(0.1, self.importance_threshold - 0.2)
                    )
                    * 255
                )
                rgb = (255 - scaled_importance, 255 - scaled_importance, 255)

            hex = "#{0:02x}{1:02x}{2:02x}".format(*self.clamp(rgb))

            if graph_node_id not in added_ids and not node.value.ignore:
                display_token = token.strip(zero_width)
                display_token = (
                    json.dumps(display_token).strip('[]"')
                    if '"' not in token
                    else display_token
                )
                if set(display_token) == {" "}:
                    display_token = f"'{display_token}'"

                fontcolor = "white" if depth != 0 and rgb[1] < 130 else "black"
                fontsize = "25" if len(display_token) < 12 else "18"
                edge_width = "7" if node.value.is_end else "3"

                current_graph.node(
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

            if (
                edge.parent is not None
                and edge.parent.id_ in visited
                and not edge.parent.value.ignore
            ):
                graph_parent_id = node_id_to_graph_id[edge.parent.id_]
                edge_tuple = (graph_parent_id, graph_node_id)
                if edge_tuple not in added_edges:
                    self.net.edge(*edge_tuple, penwidth="3", dir="back")
                    added_edges.add(edge_tuple)

        for depth, subgraph in depth_to_subgraph.items():
            self.net.subgraph(subgraph)

        path_parts = ["neuron_graphs", model_name]

        if self.folder_name is not None:
            path_parts.append(self.folder_name)

        path_parts.append(f"{self.layer}_{self.neuron}")

        save_path = base_path
        for path_part in path_parts:
            save_path += f"/{path_part}"
            if not os.path.exists(save_path):
                os.mkdir(save_path)

        filename = "graph" if graph else "trie"
        with open(f"{save_path}/{filename}", "w") as f:
            f.write(self.net.source)

    @staticmethod
    def clamp(arr):
        return [max(0, min(x, 255)) for x in arr]
