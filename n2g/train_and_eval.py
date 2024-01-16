from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray
from sklearn.model_selection import train_test_split  # type: ignore
from transformer_lens.HookedTransformer import HookedTransformer

import n2g

from . import fit
from .augmenter import AugmentationConfig, Augmenter
from .fit import FitConfig, ImportanceConfig, PruneConfig
from .neuron_model import NeuronModel
from .neuron_store import NeuronStore


def layer_index_to_name(layer_index: int, layer_ending: str) -> str:
    return f"blocks.{layer_index}.{layer_ending}"


def train_and_eval(
    model: HookedTransformer,
    layer_index: int,
    neuron_index: int,
    augmenter: Augmenter,
    samples: List[str],
    activation_matrix: NDArray[np.float32],
    layer_ending: str,
    neuron_store: NeuronStore,
    train_proportion: float = 0.5,
    fire_threshold: float = 0.5,
    random_state: int = 0,
    train_indexes: Optional[List[int]] = None,
) -> Tuple[NeuronModel, Dict[str, Any]]:
    layer = layer_index_to_name(layer_index, layer_ending)

    layer_num = int(layer.split(".")[1])
    base_max_act = float(activation_matrix[layer_num, neuron_index])

    if train_indexes is None:
        split: Tuple[List[str], List[str]] = train_test_split(  # type: ignore
            samples, train_size=train_proportion, random_state=random_state
        )
        train_samples, test_samples = split
    else:
        train_samples = [snippet for i, snippet in enumerate(samples) if i in train_indexes]
        test_samples = [snippet for i, snippet in enumerate(samples) if i not in train_indexes]

    all_train_samples = train_samples

    neuron_model = fit.fit_neuron_model(
        model,
        layer,
        layer_index,
        neuron_index,
        all_train_samples,
        augmenter,
        base_max_act,
        config=FitConfig(
            prune_config=PruneConfig(), importance_config=ImportanceConfig(), augmentation_config=AugmentationConfig()
        ),
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
