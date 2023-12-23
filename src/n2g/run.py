import json
import os
import random
import time
import traceback
from typing import List
import numpy as np
from numpy.typing import NDArray
from transformer_lens import HookedTransformer

import n2g
from n2g.fast_augmenter import FastAugmenter
from n2g.neuron_store import NeuronStore


def run_training(
    model: HookedTransformer,
    layers: List[int],
    neurons_per_layer: int,
    layer_ending: str,
    augmenter: FastAugmenter,
    activation_matrix: NDArray[np.float32],
    model_name: str,
    base_path: str,
):
    random.seed(0)

    all_neuron_indices = [i for i in range(neurons_per_layer)]

    if not os.path.exists(f"{base_path}/neuron_graphs/{model_name}"):
        os.mkdir(f"{base_path}/neuron_graphs/{model_name}")

    neuron_store = NeuronStore(
        f"{base_path}/neuron_graphs/{model_name}/neuron_store.json"
    )

    folder_path = os.path.join(base_path, f"neuron_graphs/{model_name}")

    if not os.path.exists(folder_path):
        print("Making", folder_path, flush=True)
        os.mkdir(folder_path)

    if os.path.exists(f"{folder_path}/stats.json"):
        with open(f"{folder_path}/stats.json") as ifh:
            all_stats = json.load(ifh)

    else:
        all_stats = {}

    printerval = 10

    for layer in layers:
        t1 = time.perf_counter()

        chosen_neuron_indices = all_neuron_indices

        all_stats[layer] = {}
        for i, neuron in enumerate(chosen_neuron_indices):
            print(f"{layer=} {neuron=}", flush=True)
            try:
                stats = n2g.train_and_eval(
                    model,
                    layer,
                    neuron,
                    augmenter,
                    base_path,
                    model_name,
                    activation_matrix,
                    layer_ending,
                    neuron_store,
                )

                all_stats[layer][neuron] = stats

                if i % printerval == 0:
                    t2 = time.perf_counter()
                    elapsed = t2 - t1
                    rate = printerval / elapsed
                    remaining = (len(chosen_neuron_indices) - i) / rate / 60
                    print(
                        f"{i} complete, batch took {elapsed / 60:.2f} mins, {rate=:.2f} neurons/s, {remaining=:.1f} mins"
                    )
                    t1 = t2

                    neuron_store.save()
                    with open(f"{folder_path}/stats.json", "w") as ofh:
                        json.dump(all_stats, ofh, indent=2)

            except Exception as e:
                print(traceback.format_exc(), flush=True)
                print("Failed", flush=True)

    neuron_store.save()
    with open(f"{folder_path}/stats.json", "w") as ofh:
        json.dump(all_stats, ofh, indent=2)
