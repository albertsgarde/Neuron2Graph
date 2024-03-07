from typing import Callable, List, Tuple

from jaxtyping import Float, Int
from torch import Tensor

from . import evaluate, fit
from .augmenter import Augmenter
from .fit import FitConfig
from .neuron_model import NeuronModel
from .stats import NeuronStats
from .tokenizer import Tokenizer


def train_and_eval(
    feature_activation: Callable[
        [Int[Tensor, "num_samples sample_length"]], Float[Tensor, "num_samples sample_length"]
    ],
    tokenizer: Tokenizer,
    augmenter: Augmenter,
    train_samples: List[str],
    test_samples: List[str],
    base_max_activation: float,
    fire_threshold: float,
    fit_config: FitConfig,
) -> Tuple[NeuronModel, NeuronStats]:
    neuron_model = fit.fit_neuron_model(
        feature_activation,
        tokenizer,
        train_samples,
        augmenter,
        base_max_activation,
        config=fit_config,
    )

    print("Fitted model", flush=True)

    stats = evaluate.evaluate(
        feature_activation,
        tokenizer,
        neuron_model,
        base_max_activation,
        test_samples,
        fire_threshold,
    )

    return neuron_model, stats
