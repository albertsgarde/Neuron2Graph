import math
import os
from typing import Any, List, Optional, Tuple, TypeVar, Dict
import typing
import numpy as np
from numpy.typing import NDArray
from sklearn import metrics  # type: ignore
from sklearn.model_selection import train_test_split  # type: ignore
import torch
from torch import Tensor, FloatTensor
from transformer_lens.HookedTransformer import HookedTransformer
from jaxtyping import Int

import n2g
from n2g.augmenter import Augmenter
from n2g.neuron_store import NeuronStore
from .neuron_model import NeuronModel


def layer_index_to_name(layer_index: int, layer_ending: str) -> str:
    return f"blocks.{layer_index}.{layer_ending}"


T = TypeVar("T")


def batch(arr: List[T], batch_size: int) -> List[List[T]]:
    n = math.ceil(len(arr) / batch_size)

    extras = len(arr) - (batch_size * n)
    groups: List[List[T]] = []
    group: List[T] = []
    added_extra = False
    for element in arr:
        group.append(element)
        if len(group) >= batch_size:
            if extras and not added_extra:
                extras -= 1
                added_extra = True
                continue
            groups.append(group)
            group = []
            added_extra = False

    if group:
        groups.append(group)

    return groups


def prune(
    model: HookedTransformer,
    layer: str,
    neuron: int,
    prompt: str,
    max_length: int = 1024,
    proportion_threshold: float = -0.5,
    absolute_threshold: Optional[float] = None,
    token_activation_threshold: float = 1,
    window: int = 0,
    cutoff: int = 30,
    batch_size: int = 4,
    max_post_context_tokens: int = 5,
    skip_threshold: int = 0,
    skip_interval: int = 5,
) -> List[Tuple[str, float, float]]:
    """Prune an input prompt to the shortest string that preserves x% of neuron activation on the most activating token."""

    prepend_bos = True
    tokens: Int[Tensor, "batch_pos"] = model.to_tokens(prompt, prepend_bos=prepend_bos)
    str_tokens: List[str] = model.to_str_tokens(prompt, prepend_bos=prepend_bos)  # type: ignore

    if len(tokens[0]) > max_length:
        tokens = tokens[0, :max_length].unsqueeze(0)

    _logits, cache = model.run_with_cache(tokens)  # type: ignore
    activations = cache[layer][0, :, neuron].cpu()

    full_initial_max = torch.max(activations).cpu().item()

    (
        sentences,
        _sentence_to_token_indices,
        token_to_sentence_indices,
    ) = n2g.word_tokenizer.sentence_tokenizer(str_tokens)

    strong_indices: Tensor = torch.where(  # type: ignore
        activations >= token_activation_threshold * full_initial_max
    )[0]
    strong_activations = activations[strong_indices].cpu()
    strong_indices = strong_indices.cpu()

    pruned_sentences: List[str] = []
    initial_maxes: List[float] = []
    truncated_maxes: List[float] = []

    for initial_argmax_tensor, initial_max_tensor in zip(
        strong_indices, strong_activations
    ):
        initial_argmax: int = initial_argmax_tensor.item()  # type: ignore
        initial_max: float = initial_max_tensor.item()

        max_sentence_index = token_to_sentence_indices[initial_argmax]
        relevant_str_tokens = [
            str_token
            for sentence in sentences[: max_sentence_index + 1]
            for str_token in sentence
        ]

        prior_context = relevant_str_tokens[: initial_argmax + 1]

        post_context = relevant_str_tokens[initial_argmax + 1 :]

        shortest_successful_prompt: Optional[str] = None

        truncated_prompts: List[str] = []
        added_tokens: List[int] = []

        count = 0
        full_prior = prior_context[: max(0, initial_argmax - window + 1)]

        for i in reversed(range(len(full_prior))):
            count += 1

            if count > cutoff:
                break

            if (
                not count == len(full_prior)
                and count >= skip_threshold
                and count % skip_interval != 0
            ):
                continue

            truncated_prompt = prior_context[i:]
            joined = "".join(truncated_prompt)
            truncated_prompts.append(joined)
            added_tokens.append(i)

        batched_truncated_prompts = batch(truncated_prompts, batch_size)
        batched_added_tokens = batch(added_tokens, batch_size)

        finished = False

        truncated_max: Optional[float] = None
        for i, (truncated_batch, added_tokens_batch) in enumerate(
            zip(batched_truncated_prompts, batched_added_tokens)
        ):
            truncated_tokens = model.to_tokens(truncated_batch, prepend_bos=prepend_bos)

            _logits, cache = model.run_with_cache(truncated_tokens)  # type: ignore
            all_truncated_activations = cache[layer][:, :, neuron].cpu()

            for j, truncated_activations in enumerate(all_truncated_activations):
                num_added_tokens = added_tokens_batch[j]
                truncated_argmax = (
                    torch.argmax(truncated_activations).cpu().item() + num_added_tokens
                )

                if prepend_bos:
                    truncated_argmax -= 1
                truncated_max = torch.max(truncated_activations).cpu().item()

                shortest_prompt = truncated_batch[j]

                if (
                    truncated_argmax == initial_argmax
                    and (
                        (truncated_max - initial_max) / initial_max
                        > proportion_threshold
                        or (
                            absolute_threshold is not None
                            and truncated_max >= absolute_threshold
                        )
                    )
                ) or (
                    i == len(batched_truncated_prompts) - 1
                    and j == len(all_truncated_activations) - 1
                ):
                    shortest_successful_prompt = shortest_prompt
                    finished = True
                    break

            if finished:
                break

        if shortest_successful_prompt is None:
            raise Exception("No successful prompt found")

        pruned_sentence: str = shortest_successful_prompt

        pruned_sentence += "".join(post_context[:max_post_context_tokens])

        pruned_sentences.append(pruned_sentence)
        initial_maxes.append(initial_max)
        assert truncated_max is not None, "Truncated max is None"
        truncated_maxes.append(truncated_max)

    return list(zip(pruned_sentences, initial_maxes, truncated_maxes))


def measure_importance(
    model: HookedTransformer,
    layer: str,
    neuron: int,
    prompt: str,
    initial_argmax: int | None = None,
    max_length: int = 1024,
    max_activation: float | None = None,
    masking_token: int = 1,
    threshold: float = 0.8,
    scale_factor: float = 1,
) -> Tuple[NDArray[np.float32], float, List[str], List[Tuple[str, float]], int]:
    """Compute a measure of token importance by masking each token and measuring the drop in activation on the max activating token"""

    prepend_bos = True
    tokens = model.to_tokens(prompt, prepend_bos=prepend_bos)
    str_tokens: List[str] = model.to_str_tokens(prompt, prepend_bos=prepend_bos)  # type: ignore

    if len(tokens[0]) > max_length:
        tokens = tokens[0, :max_length].unsqueeze(0)

    importances_matrix: List[NDArray[np.float32]] = []

    masked_prompts = tokens.repeat(len(tokens[0]) + 1, 1)

    for i in range(1, len(masked_prompts)):
        masked_prompts[i, i - 1] = masking_token

    _logits, cache = model.run_with_cache(masked_prompts)  # type: ignore
    activations = cache[layer][:, :, neuron].cpu()

    all_masked_activations = activations[1:, :]
    activations = activations[0, :]

    if initial_argmax is None:
        initial_argmax = typing.cast(int, torch.argmax(activations).item())
    else:
        # This could be wrong
        initial_argmax = min(initial_argmax, len(activations) - 1)

    initial_max: float = typing.cast(float, activations[initial_argmax].item())

    if max_activation is None:
        max_activation = initial_max

    tokens_and_activations: List[Tuple[str, float]] = [
        (str_token, round(activation.item() * scale_factor / max_activation, 3))
        for str_token, activation in zip(str_tokens, activations)
    ]
    important_tokens: List[str] = []
    tokens_and_importances: List[Tuple[str, float]] = [
        (str_token, 0) for str_token in str_tokens
    ]

    for i, masked_activations in enumerate(all_masked_activations):
        # Get importance of the given token for all tokens
        importances_row: List[float] = []
        for j, activation in enumerate(masked_activations):
            activation = activation.item()
            normalised_activation: float = 1 - (activation / activations[j].item())
            importances_row.append(normalised_activation)

        importances_matrix.append(np.array(importances_row))

        masked_max = masked_activations[initial_argmax].item()
        normalised_activation = 1 - (masked_max / initial_max)

        str_token, _ = tokens_and_importances[i]
        tokens_and_importances[i] = str_token, normalised_activation
        if normalised_activation >= threshold and str_token != "<|endoftext|>":
            important_tokens.append(str_token)

    # Flip so we have the importance of all tokens for a given token
    return (
        np.array(importances_matrix),
        initial_max,
        important_tokens,
        tokens_and_activations,
        initial_argmax,
    )


def evaluate(
    neuron_model: NeuronModel,
    data: List[Tuple[List[str], torch.FloatTensor]],
    fire_threshold: float,
) -> Dict[str, Any]:
    y: List[int] = []
    y_pred: List[int] = []
    y_act: NDArray[np.float32] = np.array([])
    y_pred_act: NDArray[np.float32] = np.array([])
    for prompt_tokens, activations in data:
        pred_activations = neuron_model.forward([prompt_tokens])[0]

        y_act = np.concatenate((y_act, np.array(activations)))
        y_pred_act = np.concatenate((y_pred_act, np.array(pred_activations)))

        pred_firings = [
            int(pred_activation >= fire_threshold)
            for pred_activation in pred_activations
        ]
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


def augment_and_return(
    model: HookedTransformer,
    layer: str,
    neuron: int,
    aug: Augmenter,
    pruned_prompt: str,
    base_max_act: float | None = None,
    use_index: bool = False,
    scale_factor: float = 1,
) -> List[Tuple[NDArray[np.float32], List[Tuple[str, float]]]]:
    info: List[Tuple[NDArray[np.float32], List[Tuple[str, float]]]] = []
    (
        importances_matrix,
        initial_max_act,
        important_tokens,
        tokens_and_activations,
        initial_max_index,
    ) = measure_importance(
        model,
        layer,
        neuron,
        pruned_prompt,
        max_activation=base_max_act,
        scale_factor=scale_factor,
    )

    if base_max_act is not None:
        initial_max_act = base_max_act

    positive_prompts, negative_prompts = n2g.augmenter.augment(
        model,
        layer,
        neuron,
        pruned_prompt,
        aug,
        important_tokens=set(important_tokens),
    )

    for prompt in positive_prompts:
        if use_index:
            (
                importances_matrix,
                _max_act,
                _,
                tokens_and_activations,
                _max_index,
            ) = measure_importance(
                model,
                layer,
                neuron,
                prompt,
                max_activation=initial_max_act,
                initial_argmax=initial_max_index,
                scale_factor=scale_factor,
            )
        else:
            (
                importances_matrix,
                _max_act,
                _,
                tokens_and_activations,
                _max_index,
            ) = measure_importance(
                model,
                layer,
                neuron,
                prompt,
                max_activation=initial_max_act,
                scale_factor=scale_factor,
            )
        info.append((importances_matrix, tokens_and_activations))

    for prompt in negative_prompts:
        if use_index:
            (
                importances_matrix,
                _max_act,
                _,
                tokens_and_activations,
                _max_index,
            ) = measure_importance(
                model,
                layer,
                neuron,
                prompt,
                max_activation=initial_max_act,
                initial_argmax=initial_max_index,
                scale_factor=scale_factor,
            )
        else:
            (
                importances_matrix,
                _max_act,
                _,
                tokens_and_activations,
                _max_index,
            ) = measure_importance(
                model,
                layer,
                neuron,
                prompt,
                max_activation=initial_max_act,
                scale_factor=scale_factor,
            )
        info.append((importances_matrix, tokens_and_activations))

    return info


def train_and_eval(
    model: HookedTransformer,
    layer_index: int,
    neuron: int,
    aug: Augmenter,
    graph_dir: str,
    samples: List[str],
    activation_matrix: NDArray[np.float32],
    layer_ending: str,
    neuron_store: NeuronStore,
    train_proportion: float = 0.5,
    fire_threshold: float = 0.5,
    random_state: int = 0,
    train_indexes: Optional[List[int]] = None,
) -> Dict[str, Any]:
    layer = layer_index_to_name(layer_index, layer_ending)

    layer_num = int(layer.split(".")[1])
    base_max_act = float(activation_matrix[layer_num, neuron])

    if train_indexes is None:
        split: Tuple[List[str], List[str]] = train_test_split(  # type: ignore
            samples, train_size=train_proportion, random_state=random_state
        )
        train_samples, test_samples = split
    else:
        train_samples = [
            snippet for i, snippet in enumerate(samples) if i in train_indexes
        ]
        test_samples = [
            snippet for i, snippet in enumerate(samples) if i not in train_indexes
        ]

    all_train_samples = train_samples

    all_info: List[List[Tuple[NDArray[Any], List[Tuple[str, float]]]]] = []
    for i, snippet in enumerate(all_train_samples):
        # if i % 10 == 0:
        print(f"Processing {i + 1} of {len(all_train_samples)}", flush=True)

        pruned_results = prune(
            model,
            layer,
            neuron,
            snippet,
        )

        for pruned_prompt, initial_max_act, truncated_max_act in pruned_results:
            scale_factor = initial_max_act / truncated_max_act

            info = augment_and_return(
                model,
                layer,
                neuron,
                aug,
                pruned_prompt,
                base_max_act=base_max_act,
                scale_factor=scale_factor,
            )
            all_info.append(info)

    neuron_model = NeuronModel(layer_num, neuron)
    neuron_model.fit(all_info)
    neuron_model.update_neuron_store(neuron_store)

    net = neuron_model.graphviz()

    file_path = os.path.join(graph_dir, f"{neuron_model.layer}_{neuron}")
    with open(file_path, "w") as f:
        f.write(net.source)

    print("Fitted model", flush=True)

    max_test_data: List[Tuple[List[str], FloatTensor]] = []
    for snippet in test_samples:
        tokens = model.to_tokens(snippet, prepend_bos=True)
        str_tokens: List[str] = typing.cast(
            List[str], model.to_str_tokens(snippet, prepend_bos=True)  # type: ignore
        )
        _logits, cache = model.run_with_cache(tokens)  # type: ignore
        activations: FloatTensor = cache[layer][0, :, neuron].cpu()  # type: ignore
        max_test_data.append(
            (str_tokens, typing.cast(FloatTensor, activations / base_max_act))
        )

    print("Max Activating Evaluation Data", flush=True)
    try:
        stats = evaluate(neuron_model, max_test_data, fire_threshold)
    except Exception as e:
        stats = {}
        print(f"Stats failed with error: {e}", flush=True)

    return stats
