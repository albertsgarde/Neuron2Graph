from collections import Counter, defaultdict
import json
import os
from typing import Any, Dict


class NeuronStore:
    def __init__(self, path: str):
        if not os.path.exists(path):
            neuron_store: Dict[str, Dict[Any, Any]] = {
                "activating": {},
                "important": {},
            }
            with open(path, "w") as ofh:
                json.dump(neuron_store, ofh, indent=2, ensure_ascii=False)

        with open(path) as ifh:
            self.store = json.load(ifh)

        self._to_sets()
        self.path = path
        self._count_tokens()
        self._by_neuron()

    def save(self):
        self._to_lists()
        with open(self.path, "w") as ofh:
            json.dump(self.store, ofh, indent=2, ensure_ascii=False)
        self._to_sets()

    def _to_sets(self):
        self.store = {
            token_type: {token: set(info) for token, info in token_dict.items()}
            for token_type, token_dict in self.store.items()
        }

    def _to_lists(self):
        self.store = {
            token_type: {token: list(set(info)) for token, info in token_dict.items()}
            for token_type, token_dict in self.store.items()
        }

    def _by_neuron(self):
        self.neuron_to_tokens = {}
        for token_type, token_dict in self.store.items():
            for token, neurons in token_dict.items():
                for neuron in neurons:
                    if neuron not in self.neuron_to_tokens:
                        self.neuron_to_tokens[neuron] = {
                            "activating": set(),
                            "important": set(),
                        }
                    self.neuron_to_tokens[neuron][token_type].add(token)

    def _count_tokens(self):
        self.neuron_individual_token_counts = defaultdict(Counter)
        self.neuron_total_token_counts = Counter()
        for token_type, token_dict in self.store.items():
            for token, neurons in token_dict.items():
                for neuron in neurons:
                    self.neuron_individual_token_counts[neuron][token] += 1
                    self.neuron_total_token_counts[neuron] += 1
