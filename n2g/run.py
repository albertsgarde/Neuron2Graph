import json
import os
import random
import traceback
import typing
from typing import Any, Dict, List, Tuple

import numpy as np
from numpy.typing import NDArray
from torch import device
from transformer_lens.HookedTransformer import HookedTransformer
from transformers import AutoModelForMaskedLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizer  # type: ignore

import n2g
from n2g.augmenter import Augmenter
from n2g.neuron_store import NeuronStore

from .word_tokenizer import WordTokenizer


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
) -> Tuple[NeuronStore, Dict[int, Dict[int, Dict[Any, Any]]]]:
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

    return neuron_store, all_stats


def run(
    model_name: str,
    layer_indices: List[int],
    neuron_indices: List[int],
    layer_ending: str,
    activation_matrix: NDArray[np.float32],
    word_to_casings: Any,
    aug_model_name: str,
    output_dir: str,
    device: device,
) -> Tuple[NeuronStore, Dict[int, Dict[int, Dict[Any, Any]]]]:
    model: HookedTransformer = HookedTransformer.from_pretrained(model_name).to(device)  # type: ignore

    aug_model: PreTrainedModel = typing.cast(
        PreTrainedModel,
        AutoModelForMaskedLM.from_pretrained(aug_model_name).to(  # type: ignore
            device
        ),
    )
    aug_tokenizer: PreTrainedTokenizer = typing.cast(
        PreTrainedTokenizer,
        AutoTokenizer.from_pretrained(aug_model_name),  # type: ignore
    )

    stick_tokens = {"'"}
    word_tokenizer = WordTokenizer(set(), stick_tokens)
    augmenter = Augmenter(aug_model, aug_tokenizer, word_tokenizer, word_to_casings)

    return run_training(
        model, layer_indices, neuron_indices, layer_ending, augmenter, activation_matrix, model_name, output_dir
    )
