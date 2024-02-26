from typing import Callable, List, Tuple

from jaxtyping import Float, Int
from torch import Tensor
from transformer_lens import HookedTransformer  # type: ignore[import]

import n2g
from n2g.stats import NeuronStats  # type: ignore

from . import fit
from .augmenter import Augmenter
from .fit import FitConfig
from .neuron_model import NeuronModel
from .neuron_store import NeuronStore


def train_and_eval(
    model: HookedTransformer,
    neuron_activation: Callable[[Int[Tensor, "num_samples sample_length"]], Float[Tensor, "num_samples sample_length"]],
    layer_index: int,
    neuron_index: int,
    augmenter: Augmenter,
    train_samples: List[str],
    test_samples: List[str],
    base_max_activation: float,
    neuron_store: NeuronStore,
    fire_threshold: float,
    fit_config: FitConfig,
) -> Tuple[NeuronModel, NeuronStats]:
    neuron_model = fit.fit_neuron_model(
        model,
        neuron_activation,
        train_samples,
        augmenter,
        base_max_activation,
        config=fit_config,
    )
    neuron_model.update_neuron_store(neuron_store, str(layer_index), neuron_index)

    print("Fitted model", flush=True)

    stats = n2g.evaluate(
        model,
        neuron_activation,
        neuron_index,
        neuron_model,
        base_max_activation,
        test_samples,
        fire_threshold,
    )

    return neuron_model, stats
