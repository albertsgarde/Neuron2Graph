from typing import Callable, List

import numpy as np
from jaxtyping import Bool, Float, Int
from numpy.typing import NDArray
from torch import Tensor

from .feature_model import FeatureModel
from .stats import NeuronStats
from .tokenizer import Tokenizer


def evaluate(
    feature_activation: Callable[
        [Int[Tensor, "num_samples sample_length"]], Float[Tensor, "num_samples sample_length"]
    ],
    tokenizer: Tokenizer,
    neuron_model: FeatureModel,
    base_max_act: float,
    test_samples: List[str],
    fire_threshold: float,
) -> NeuronStats:
    # Prepending BOS is unnecessary since they are already prepended.
    test_tokens: Int[Tensor, "num_samples sample_length"]
    test_str_tokens: list[list[str]]
    test_tokens, test_str_tokens = tokenizer.batch_tokenize_with_str(test_samples, prepend_bos=False)

    pred_activations: Float[NDArray, "num_samples sample_length"] = np.full(test_tokens.shape, float("nan"))

    # Get real activations
    activations = (feature_activation(test_tokens) / base_max_act).cpu().numpy()

    assert not np.isnan(activations).any(), "activations should not contain NaNs"
    assert not np.isinf(activations).any(), "activations should not contain Infs"

    # Get predicted activations
    for sample_index, (sample_tokens, _sample_str_tokens) in enumerate(zip(test_tokens, test_str_tokens)):
        pred_sample_activations = neuron_model.forward_tokens([[token for token in sample_tokens]])[0]
        pred_activations[sample_index, :] = np.array(pred_sample_activations)

    assert not np.isnan(pred_activations).any(), "pred_activations should not contain NaNs"
    assert not np.isinf(pred_activations).any(), "pred_activations should not contain Infs"

    firings: Bool[NDArray, " num_samples*sample_length"] = (activations >= fire_threshold).ravel()
    pred_firings: Bool[NDArray, " num_samples*sample_length"] = (pred_activations >= fire_threshold).ravel()

    return NeuronStats.from_firings(firings, pred_firings)
