from typing import Callable, List, Tuple

from jaxtyping import Float, Int
from torch import Tensor

import n2g

from . import fit
from .augmenter import Augmenter
from .fit import FitConfig
from .neuron_model import NeuronModel
from .neuron_store import NeuronStore
from .stats import NeuronStats
from .tokenizer import Tokenizer


def train_and_eval(
    neuron_activation: Callable[[Int[Tensor, "num_samples sample_length"]], Float[Tensor, "num_samples sample_length"]],
    tokenizer: Tokenizer,
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
        neuron_activation,
        tokenizer,
        train_samples,
        augmenter,
        base_max_activation,
        config=fit_config,
    )
    neuron_model.update_neuron_store(neuron_store, str(layer_index), neuron_index)

    print("Fitted model", flush=True)

    stats = n2g.evaluate(
        neuron_activation,
        tokenizer,
        neuron_model,
        base_max_activation,
        test_samples,
        fire_threshold,
    )

    return neuron_model, stats
