from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Set

from n2g.feature_model import FeatureModel


class NeuronStore:
    def __init__(self) -> None:
        self._activating: Dict[str, Set[str]] = {}
        self._important: Dict[str, Set[str]] = {}

    @staticmethod
    def _store_to_sets(store: Dict[str, List[str]]) -> Dict[str, Set[str]]:
        return {token: set(indices) for token, indices in store.items()}

    @staticmethod
    def _store_to_lists(store: Dict[str, Set[str]]) -> Dict[str, List[str]]:
        return {token: list(indices) for token, indices in store.items()}

    @staticmethod
    def load(path: Path) -> NeuronStore:
        with open(path) as file:
            store = json.load(file)
            activating: Dict[str, List[str]] = store["activating"]
            important: Dict[str, List[str]] = store["important"]

            result = NeuronStore()
            result._activating = NeuronStore._store_to_sets(activating)
            result._important = NeuronStore._store_to_sets(important)
            return result

    def save(self, path: Path) -> None:
        with open(path, "w") as ofh:
            store = {
                "activating": NeuronStore._store_to_lists(self._activating),
                "important": NeuronStore._store_to_lists(self._important),
            }
            json.dump(store, ofh, indent=2, ensure_ascii=False)

    def update(self, neuron_id: str, model: FeatureModel) -> None:
        for token, activating in model.tokens():
            self.add_neuron(activating, token, neuron_id)

    def add_neuron(self, activating: bool, token: str, neuron: str) -> None:
        store = self._activating if activating else self._important
        if token not in store.keys():
            store[token] = set()
        store[token].add(neuron)
