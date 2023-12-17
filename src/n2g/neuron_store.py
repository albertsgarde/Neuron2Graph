from collections import Counter, defaultdict
import json
import os


class NeuronStore:
    def __init__(self, path):
        if not os.path.exists(path):
            neuron_store = {"activating": {}, "important": {}}
            with open(path, "w") as ofh:
                json.dump(neuron_store, ofh, indent=2, ensure_ascii=False)

        with open(path) as ifh:
            self.store = json.load(ifh)

        self.to_sets()
        self.path = path
        self.count_tokens()
        self.by_neuron()

    def save(self):
        self.to_lists()
        with open(self.path, "w") as ofh:
            json.dump(self.store, ofh, indent=2, ensure_ascii=False)
        self.to_sets()

    def to_sets(self):
        self.store = {
            token_type: {token: set(info) for token, info in token_dict.items()}
            for token_type, token_dict in self.store.items()
        }

    def to_lists(self):
        self.store = {
            token_type: {token: list(set(info)) for token, info in token_dict.items()}
            for token_type, token_dict in self.store.items()
        }

    def by_neuron(self):
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

    def search(self, tokens_and_types):
        match_arr = []

        for token, token_type in tokens_and_types:
            token_types = (
                [token_type] if token_type is not None else ["activating", "important"]
            )
            token_matches = set()

            for token_type in token_types:
                matches = self.store[token_type].get(token, set())
                token_matches |= matches

            match_arr.append(token_matches)

        valid_matches = set.intersection(*match_arr)
        return valid_matches

    def count_tokens(self):
        self.neuron_individual_token_counts = defaultdict(Counter)
        self.neuron_total_token_counts = Counter()
        for token_type, token_dict in self.store.items():
            for token, neurons in token_dict.items():
                for neuron in neurons:
                    self.neuron_individual_token_counts[neuron][token] += 1
                    self.neuron_total_token_counts[neuron] += 1

    def find_similar(self, target_token_types=None, threshold=0.9):
        if target_token_types is None:
            target_token_types = {"activating", "important"}

        similar_pairs = []
        subset_pairs = []

        for i, (neuron_1, neuron_dict_1) in enumerate(self.neuron_to_tokens.items()):
            if i % 1000 == 0:
                print(
                    f"{i} of {len(self.neuron_to_tokens.items())} complete", flush=True
                )

            for j, (neuron_2, neuron_dict_2) in enumerate(
                self.neuron_to_tokens.items()
            ):
                if i <= j:
                    continue

                all_similar = []
                all_subset = []

                for token_type in target_token_types:
                    length_1 = len(neuron_dict_1[token_type])
                    length_2 = len(neuron_dict_2[token_type])

                    intersection = neuron_dict_1[token_type] & neuron_dict_2[token_type]
                    similar = (
                        len(intersection) / max(length_1, length_2, 1)
                    ) >= threshold
                    subset = (
                        len(intersection) / max(min(length_1, length_2), 1) >= threshold
                    )

                    all_similar.append(similar)
                    all_subset.append(subset)

                if all(all_similar):
                    similar_pairs.append((neuron_1, neuron_2))
                elif all(all_subset):
                    # The first token indicates the superset neuron and the second the subset neuron
                    subset_pair = (
                        (neuron_1, neuron_2)
                        if length_2 < length_1
                        else (neuron_2, neuron_1)
                    )
                    subset_pairs.append(subset_pair)

        return similar_pairs, subset_pairs
