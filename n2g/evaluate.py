from typing import Any, Dict, List

import numpy as np
import torch
from jaxtyping import Bool, Float, Int
from numpy.typing import NDArray
from sklearn import metrics  # type: ignore
from torch import Tensor
from transformer_lens import HookedTransformer  # type: ignore[import]
from transformer_lens.hook_points import HookPoint  # type: ignore[import]

from n2g.stats import NeuronStats  # type: ignore[import]

from .neuron_model import NeuronModel


def evaluate(
    model: HookedTransformer,
    layer: str,
    neuron_index: int,
    neuron_model: NeuronModel,
    base_max_act: float,
    test_samples: List[str],
    fire_threshold: float,
) -> NeuronStats:
    test_tokens: Int[Tensor, "num_samples sample_length"] = model.to_tokens(test_samples)
    test_str_tokens: List[List[str]] = [model.tokenizer.batch_decode(sample) for sample in test_tokens]

    activations: Float[NDArray, "num_samples sample_length"] = np.zeros(test_tokens.shape)
    pred_activations: Float[NDArray, "num_samples sample_length"] = np.full(test_tokens.shape, float("nan"))

    def hook(activation: Float[Tensor, "num_samples sample_length neurons_per_layer"], hook: HookPoint) -> None:
        activations[:] = (activation[:, :, neuron_index] / base_max_act).cpu().numpy()

    assert not np.isnan(pred_activations).any(), "pred_activations should not contain NaNs"

    with torch.no_grad():
        model.run_with_hooks(test_tokens, fwd_hooks=[(layer, hook)])

    for sample_index, sample_str_tokens in enumerate(test_str_tokens):
        pred_sample_activations = neuron_model.forward([sample_str_tokens])[0]
        pred_activations[sample_index, :] = np.array(pred_sample_activations)

    firings: Bool[NDArray, "num_samples sample_length"] = activations >= fire_threshold
    pred_firings: Bool[NDArray, "num_samples sample_length"] = pred_activations >= fire_threshold

    report: Dict[str, Any] = metrics.classification_report(
        firings.ravel(), pred_firings.ravel(), target_names=["non_firing", "firing"], output_dict=True
    )  # type: ignore

    act_diff = pred_activations - activations
    mse: float = np.mean(np.power(act_diff, 2))  # type: ignore
    variance: float = np.var(activations)  # type: ignore
    correlation: float = 1.0 - (mse / variance)

    return NeuronStats.from_metrics_classification_report(report, correlation)
