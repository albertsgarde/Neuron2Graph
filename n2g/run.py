import json
import os
import random
import traceback
from typing import Any, Dict, List, Tuple

import numpy as np
from numpy.typing import NDArray
from transformer_lens.HookedTransformer import HookedTransformer

import n2g
from n2g.augmenter import Augmenter
from n2g.neuron_store import NeuronStore


def setup_paths(output_dir: str) -> Tuple[str, str, str]:
    graph_dir = os.path.join(output_dir, "graphs")
    neuron_store_path = os.path.join(output_dir, "neuron_store.json")
    stats_path = os.path.join(output_dir, "stats.json")
    # This ensures that both the base output path and the graph directory exist
    if not os.path.exists(graph_dir):
        os.makedirs(graph_dir)
    return graph_dir, neuron_store_path, stats_path


def get_stats(stats_path: str) -> Dict[int, Dict[int, Dict[Any, Any]]]:
    if os.path.exists(stats_path):
        with open(stats_path) as ifh:
            return json.load(ifh)
    else:
        return {}


def run_training(
    model: HookedTransformer,
    layer_indices: List[int],
    neuron_indices: List[int],
    layer_ending: str,
    augmenter: Augmenter,
    activation_matrix: NDArray[np.float32],
    model_name: str,
    output_dir: str,
):
    random.seed(0)

    graph_dir, neuron_store_path, stats_path = setup_paths(output_dir)

    neuron_store = NeuronStore.load(neuron_store_path) if os.path.exists(neuron_store_path) else NeuronStore()

    all_stats = get_stats(stats_path)

    for layer_index in layer_indices:
        all_stats[layer_index] = {}
        for neuron_index in neuron_indices:
            print(f"{layer_index=} {neuron_index=}", flush=True)
            try:
                training_samples = n2g.scrape_neuroscope_samples(model_name, layer_index, neuron_index)

                stats = n2g.train_and_eval(
                    model,
                    layer_index,
                    neuron_index,
                    augmenter,
                    graph_dir,
                    training_samples,
                    activation_matrix,
                    layer_ending,
                    neuron_store,
                )

                all_stats[layer_index][neuron_index] = stats

            except Exception:
                print(traceback.format_exc(), flush=True)
                print("Failed", flush=True)

    neuron_store.save(neuron_store_path)
    with open(stats_path, "w") as ofh:
        json.dump(all_stats, ofh, indent=2)
