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

from .augmenter import Augmenter
from .neuron_model import NeuronModel
from .neuron_store import NeuronStore
from .word_tokenizer import WordTokenizer


def run_training(
    model: HookedTransformer,
    layer_indices: List[int],
    neuron_indices: List[int],
    layer_ending: str,
    augmenter: Augmenter,
    activation_matrix: NDArray[np.float32],
    model_name: str,
    neuron_store: NeuronStore,
    all_stats: Dict[int, Dict[int, Dict[str, Any]]],
) -> Tuple[Dict[int, Dict[int, NeuronModel]], NeuronStore, Dict[int, Dict[int, Dict[str, Any]]]]:
    random.seed(0)

    neuron_models: Dict[int, Dict[int, NeuronModel]] = {}

    for layer_index in layer_indices:
        all_stats[layer_index] = {}
        neuron_models[layer_index] = {}
        for neuron_index in neuron_indices:
            print(f"{layer_index=} {neuron_index=}", flush=True)
            try:
                training_samples = n2g.scrape_neuroscope_samples(model_name, layer_index, neuron_index)

                neuron_model, stats = n2g.train_and_eval(
                    model,
                    layer_index,
                    neuron_index,
                    augmenter,
                    training_samples,
                    activation_matrix,
                    layer_ending,
                    neuron_store,
                )

                neuron_models[layer_index][neuron_index] = neuron_model
                all_stats[layer_index][neuron_index] = stats

            except Exception:
                print(traceback.format_exc(), flush=True)
                print("Failed", flush=True)

    return neuron_models, neuron_store, all_stats


def run(
    model_name: str,
    layer_indices: List[int],
    neuron_indices: List[int],
    layer_ending: str,
    activation_matrix: NDArray[np.float32],
    word_to_casings: Any,
    aug_model_name: str,
    neuron_store: NeuronStore,
    all_stats: Dict[int, Dict[int, Dict[str, Any]]],
    device: device,
) -> Tuple[Dict[int, Dict[int, NeuronModel]], NeuronStore, Dict[int, Dict[int, Dict[str, Any]]]]:
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
    augmenter = Augmenter(aug_model, aug_tokenizer, word_tokenizer, word_to_casings, device)

    return run_training(
        model,
        layer_indices,
        neuron_indices,
        layer_ending,
        augmenter,
        activation_matrix,
        model_name,
        neuron_store,
        all_stats,
    )
