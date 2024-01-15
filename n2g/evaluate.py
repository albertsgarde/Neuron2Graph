from typing import Any, Dict, List, Tuple

import numpy as np
from jaxtyping import Float
from numpy.typing import NDArray
from sklearn import metrics  # type: ignore
from torch import Tensor
from transformer_lens.HookedTransformer import HookedTransformer

from .neuron_model import NeuronModel


def evaluate(
    model: HookedTransformer,
    layer: str,
    neuron_index: int,
    neuron_model: NeuronModel,
    base_max_act: float,
    test_samples: List[str],
    fire_threshold: float,
) -> Dict[str, Any]:
    max_test_data: List[Tuple[List[str], Float[Tensor, ""]]] = []
    for snippet in test_samples:
        tokens = model.to_tokens(snippet, prepend_bos=True)
        str_tokens: List[str] = model.to_str_tokens(snippet, prepend_bos=True)  # type: ignore
        _logits, cache = model.run_with_cache(tokens)  # type: ignore
        activations: Float[Tensor, ""] = cache[layer][0, :, neuron_index].cpu()  # type: ignore
        max_test_data.append((str_tokens, activations / base_max_act))

    print("Max Activating Evaluation Data", flush=True)

    y: List[int] = []
    y_pred: List[int] = []
    y_act: NDArray[np.float32] = np.array([])
    y_pred_act: NDArray[np.float32] = np.array([])
    for prompt_tokens, activations in max_test_data:
        pred_activations = neuron_model.forward([prompt_tokens])[0]

        y_act = np.concatenate((y_act, np.array(activations)))
        y_pred_act = np.concatenate((y_pred_act, np.array(pred_activations)))

        pred_firings = [int(pred_activation >= fire_threshold) for pred_activation in pred_activations]
        firings = [int(activation >= fire_threshold) for activation in activations]
        y_pred.extend(pred_firings)
        y.extend(firings)

    print(metrics.classification_report(y, y_pred), flush=True)  # type: ignore
    report: Dict[str, Any] = metrics.classification_report(y, y_pred, output_dict=True)  # type: ignore

    act_diff = y_pred_act - y_act
    mse: float = np.mean(np.power(act_diff, 2))  # type: ignore
    variance: float = np.var(y_act)  # type: ignore
    correlation: float = 1.0 - (mse / variance)

    report["correlation"] = correlation
    return report
