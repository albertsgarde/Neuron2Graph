from typing import Any, Callable, Dict, List

import numpy as np
from jaxtyping import Bool, Float, Int
from numpy.typing import NDArray
from sklearn import metrics  # type: ignore
from torch import Tensor
from transformer_lens import HookedTransformer  # type: ignore[import]

from n2g.stats import NeuronStats  # type: ignore[import]

from .neuron_model import NeuronModel


def evaluate(
    model: HookedTransformer,
    neuron_activation: Callable[[Int[Tensor, "num_samples sample_length"]], Float[Tensor, "num_samples sample_length"]],
    neuron_index: int,
    neuron_model: NeuronModel,
    base_max_act: float,
    test_samples: List[str],
    fire_threshold: float,
) -> NeuronStats:
    # Prepending BOS is unnecessary since they are already prepended.
    test_tokens: Int[Tensor, "num_samples sample_length"] = model.to_tokens(test_samples, prepend_bos=False)
    test_str_tokens: List[List[str]] = [model.tokenizer.batch_decode(sample) for sample in test_tokens]

    pred_activations: Float[NDArray, "num_samples sample_length"] = np.full(test_tokens.shape, float("nan"))

    activations = (neuron_activation(test_tokens) / base_max_act).cpu().numpy()

    assert not np.isnan(activations).any(), "activations should not contain NaNs"
    assert not np.isinf(activations).any(), "activations should not contain Infs"

    for sample_index, sample_str_tokens in enumerate(test_str_tokens):
        pred_sample_activations = neuron_model.forward([sample_str_tokens])[0]
        pred_activations[sample_index, :] = np.array(pred_sample_activations)

    assert not np.isnan(pred_activations).any(), "pred_activations should not contain NaNs"
    assert not np.isinf(pred_activations).any(), "pred_activations should not contain Infs"

    firings: Bool[NDArray, "num_samples sample_length"] = activations >= fire_threshold
    pred_firings: Bool[NDArray, "num_samples sample_length"] = pred_activations >= fire_threshold

    report: Dict[str, Any] = metrics.classification_report(
        firings.ravel(), pred_firings.ravel(), target_names=["non_firing", "firing"], output_dict=True
    )  # type: ignore

    return NeuronStats.from_metrics_classification_report(report)
