from typing import Any, Dict, List, Tuple

import numpy as np
from numpy.typing import NDArray
from transformer_lens.HookedTransformer import HookedTransformer

import n2g

from . import fit
from .augmenter import Augmenter
from .fit import FitConfig
from .neuron_model import NeuronModel
from .neuron_store import NeuronStore


def layer_index_to_name(layer_index: int, layer_ending: str) -> str:
    return f"blocks.{layer_index}.{layer_ending}"


def train_and_eval(
    model: HookedTransformer,
    layer_index: int,
    neuron_index: int,
    augmenter: Augmenter,
    train_samples: List[str],
    test_samples: List[str],
    activation_matrix: NDArray[np.float32],
    layer_ending: str,
    neuron_store: NeuronStore,
    fire_threshold: float,
    fit_config: FitConfig,
) -> Tuple[NeuronModel, Dict[str, Any]]:
    layer = layer_index_to_name(layer_index, layer_ending)

    layer_num = int(layer.split(".")[1])
    base_max_act = float(activation_matrix[layer_num, neuron_index])

    neuron_model = fit.fit_neuron_model(
        model,
        layer,
        layer_index,
        neuron_index,
        train_samples,
        augmenter,
        base_max_act,
        config=fit_config,
    )
    neuron_model.update_neuron_store(neuron_store)

    print("Fitted model", flush=True)

    try:
        stats = n2g.evaluate(
            model,
            layer,
            neuron_index,
            neuron_model,
            base_max_act,
            test_samples,
            fire_threshold,
        )
    except Exception as e:
        stats = {}
        print(f"Stats failed with error: {e}", flush=True)

    return neuron_model, stats
