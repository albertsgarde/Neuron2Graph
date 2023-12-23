import json
import math
import re
from typing import Any, List, Tuple
import numpy as np
from numpy.typing import NDArray
import requests
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
import torch

import n2g
from n2g.neuron_store import NeuronStore
from .neuron_model import NeuronModel


parser = re.compile('\{"tokens": ')


def get_snippets(model_name, layer, neuron):
    """Get the max activating dataset examples for a given neuron in a model"""
    base_url = f"https://neuroscope.io/{model_name}/{layer}/{neuron}.html"

    response = requests.get(base_url)
    webpage = response.text

    parts = parser.split(webpage)
    snippets = []
    for i, part in enumerate(parts):
        if i == 0 or i % 2 != 0:
            continue

        token_str = part.split(', "values": ')[0]

        tokens = json.loads(token_str)

        snippet = "".join(tokens)

        snippets.append(snippet)

    if len(snippets) != 20:
        raise Exception
    return snippets


def layer_index_to_name(layer_index, layer_ending):
    return f"blocks.{layer_index}.{layer_ending}"


def batch(arr, n=None, batch_size=None):
    if n is None and batch_size is None:
        raise ValueError("Either n or batch_size must be provided")
    if n is not None and batch_size is not None:
        raise ValueError("Either n or batch_size must be provided, not both")

    if n is not None:
        batch_size = math.floor(len(arr) / n)
    elif batch_size is not None:
        n = math.ceil(len(arr) / batch_size)

    extras = len(arr) - (batch_size * n)
    groups = []
    group = []
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


def fast_prune(
    model,
    layer,
    neuron,
    prompt,
    max_length=1024,
    proportion_threshold=-0.5,
    absolute_threshold=None,
    token_activation_threshold=0.75,
    window=0,
    return_maxes=False,
    cutoff=30,
    batch_size=4,
    max_post_context_tokens=5,
    skip_threshold=0,
    skip_interval=5,
    return_intermediates=False,
    **kwargs,
):
    """Prune an input prompt to the shortest string that preserves x% of neuron activation on the most activating token."""

    prepend_bos = True
    tokens = model.to_tokens(prompt, prepend_bos=prepend_bos)
    str_tokens = model.to_str_tokens(prompt, prepend_bos=prepend_bos)

    if len(tokens[0]) > max_length:
        tokens = tokens[0, :max_length].unsqueeze(0)

    logits, cache = model.run_with_cache(tokens)
    activations = cache[layer][0, :, neuron].cpu()

    full_initial_max = torch.max(activations).cpu().item()

    (
        sentences,
        sentence_to_token_indices,
        token_to_sentence_indices,
    ) = n2g.word_tokenizer.sentence_tokenizer(str_tokens)

    strong_indices = torch.where(
        activations >= token_activation_threshold * full_initial_max
    )[0]
    strong_activations = activations[strong_indices].cpu()
    strong_indices = strong_indices.cpu()

    strong_sentence_indices = [
        token_to_sentence_indices[index.item()] for index in strong_indices
    ]

    pruned_sentences = []
    final_max_indices = []
    all_intermediates = []
    initial_maxes = []
    truncated_maxes = []

    for strong_sentence_index, initial_argmax, initial_max in zip(
        strong_sentence_indices, strong_indices, strong_activations
    ):
        initial_argmax = initial_argmax.item()
        initial_max = initial_max.item()

        max_sentence_index = token_to_sentence_indices[initial_argmax]
        relevant_str_tokens = [
            str_token
            for sentence in sentences[: max_sentence_index + 1]
            for str_token in sentence
        ]

        prior_context = relevant_str_tokens[: initial_argmax + 1]

        post_context = relevant_str_tokens[initial_argmax + 1 :]

        shortest_successful_prompt = None
        final_max_index = None

        truncated_prompts = []
        added_tokens = []

        count = 0
        full_prior = prior_context[: max(0, initial_argmax - window + 1)]

        for i, str_token in reversed(list(enumerate(full_prior))):
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

        batched_truncated_prompts = batch(truncated_prompts, batch_size=batch_size)
        batched_added_tokens = batch(added_tokens, batch_size=batch_size)

        finished = False
        intermediates = []
        for i, (truncated_batch, added_tokens_batch) in enumerate(
            zip(batched_truncated_prompts, batched_added_tokens)
        ):
            truncated_tokens = model.to_tokens(truncated_batch, prepend_bos=prepend_bos)

            logits, cache = model.run_with_cache(truncated_tokens)
            all_truncated_activations = cache[layer][:, :, neuron].cpu()

            for j, truncated_activations in enumerate(all_truncated_activations):
                num_added_tokens = added_tokens_batch[j]
                truncated_argmax = (
                    torch.argmax(truncated_activations).cpu().item() + num_added_tokens
                )
                final_max_index = torch.argmax(truncated_activations).cpu().item()

                if prepend_bos:
                    truncated_argmax -= 1
                    final_max_index -= 1
                truncated_max = torch.max(truncated_activations).cpu().item()

                shortest_prompt = truncated_batch[j]

                if not shortest_prompt.startswith("<|endoftext|>"):
                    truncated_str_tokens = model.to_str_tokens(
                        truncated_batch[j], prepend_bos=False
                    )
                    intermediates.append(
                        (shortest_prompt, truncated_str_tokens[0], truncated_max)
                    )

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

        pruned_sentence = "".join(shortest_successful_prompt)

        if max_post_context_tokens is not None:
            pruned_sentence += "".join(post_context[:max_post_context_tokens])

        pruned_sentences.append(pruned_sentence)
        final_max_indices.append(final_max_index)
        initial_maxes.append(initial_max)
        truncated_maxes.append(truncated_max)
        all_intermediates.append(intermediates)

    if return_maxes:
        return list(
            zip(pruned_sentences, final_max_indices, initial_maxes, truncated_maxes)
        )

    elif return_intermediates:
        return list(zip(pruned_sentences, all_intermediates))

    return list(zip(pruned_sentences, final_max_indices))


def fast_measure_importance(
    model,
    layer: int,
    neuron: int,
    prompt,
    initial_argmax=None,
    max_length: int = 1024,
    max_activation=None,
    masking_token=1,
    threshold: float = 0.8,
    scale_factor=1,
    activation_threshold: float = 0.1,
) -> Tuple[NDArray[Any], float, List[str], List[List[str | float]], int]:
    """Compute a measure of token importance by masking each token and measuring the drop in activation on the max activating token"""

    prepend_bos = True
    tokens = model.to_tokens(prompt, prepend_bos=prepend_bos)
    str_tokens: List[str] = model.to_str_tokens(prompt, prepend_bos=prepend_bos)

    if len(tokens[0]) > max_length:
        tokens = tokens[0, :max_length].unsqueeze(0)

    importances_matrix: List[NDArray[Any]] = []

    masked_prompts = tokens.repeat(len(tokens[0]) + 1, 1)

    for i in range(1, len(masked_prompts)):
        masked_prompts[i, i - 1] = masking_token

    logits, cache = model.run_with_cache(masked_prompts)
    all_masked_activations = cache[layer][1:, :, neuron].cpu()

    activations = cache[layer][0, :, neuron].cpu()

    if initial_argmax is None:
        initial_argmax = torch.argmax(activations).cpu().item()
    else:
        # This could be wrong
        initial_argmax = min(initial_argmax, len(activations) - 1)

    initial_max: float = activations[initial_argmax].cpu().item()

    if max_activation is None:
        max_activation = initial_max
    scale = min(1, initial_max / max_activation)

    tokens_and_activations: List[List[str | float]] = [
        [str_token, round(activation.cpu().item() * scale_factor / max_activation, 3)]
        for str_token, activation in zip(str_tokens, activations)
    ]
    important_tokens: List[str] = []
    tokens_and_importances: List[Tuple[str, float]] = [
        (str_token, 0) for str_token in str_tokens
    ]

    for i, masked_activations in enumerate(all_masked_activations):
        # Get importance of the given token for all tokens
        importances_row = []
        for j, activation in enumerate(masked_activations):
            activation = activation.cpu().item()
            normalised_activation: float = 1 - (
                activation / activations[j].cpu().item()
            )
            importances_row.append((str_tokens[j], normalised_activation))

        importances_matrix.append(np.array(importances_row))

        masked_max = masked_activations[initial_argmax].cpu().item()
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


def evaluate(neuron_model, data, fire_threshold: float = 0.5, **kwargs):
    y = []
    y_pred = []
    y_act = []
    y_pred_act = []
    for prompt_tokens, activations in data:
        non_zero_indices = [
            i for i, activation in enumerate(activations) if activation > 0
        ]
        start = max(0, non_zero_indices[0] - 10)
        end = min(len(prompt_tokens) - 1, non_zero_indices[-1] + 10)
        pred_activations = neuron_model.forward([prompt_tokens])[0]

        y_act.extend(activations)
        y_pred_act.extend(pred_activations)

        important_context = list(zip(prompt_tokens, activations, pred_activations))[
            start:end
        ]

        pred_firings = [
            int(pred_activation >= fire_threshold)
            for pred_activation in pred_activations
        ]
        firings = [int(activation >= fire_threshold) for activation in activations]
        y_pred.extend(pred_firings)
        y.extend(firings)

    print(classification_report(y, y_pred), flush=True)
    report = classification_report(y, y_pred, output_dict=True)

    y_act = np.array(y_act)
    y_pred_act = np.array(y_pred_act)

    act_diff = y_pred_act - y_act
    mse = np.mean(np.power(act_diff, 2))
    variance = np.var(y_act)
    correlation = 1 - (mse / variance)

    report["correlation"] = correlation
    return report


def augment_and_return(
    model,
    layer,
    neuron,
    aug,
    pruned_prompt,
    base_max_act=None,
    use_index=False,
    scale_factor=1,
    **kwargs,
) -> List[Tuple[NDArray[Any], List[List[str | float]], int]]:
    info: List[Tuple[NDArray[Any], List[List[str | float]], int]] = []
    (
        importances_matrix,
        initial_max_act,
        important_tokens,
        tokens_and_activations,
        initial_max_index,
    ) = fast_measure_importance(
        model,
        layer,
        neuron,
        pruned_prompt,
        max_activation=base_max_act,
        scale_factor=scale_factor,
    )

    if base_max_act is not None:
        initial_max_act = base_max_act

    positive_prompts, negative_prompts = n2g.fast_augmenter.augment(
        model,
        layer,
        neuron,
        pruned_prompt,
        aug,
        important_tokens=set(important_tokens),
        **kwargs,
    )

    for i, (prompt, activation, change) in enumerate(positive_prompts):
        if use_index:
            (
                importances_matrix,
                max_act,
                _,
                tokens_and_activations,
                max_index,
            ) = fast_measure_importance(
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
                max_act,
                _,
                tokens_and_activations,
                max_index,
            ) = fast_measure_importance(
                model,
                layer,
                neuron,
                prompt,
                max_activation=initial_max_act,
                scale_factor=scale_factor,
            )
        info.append((importances_matrix, tokens_and_activations, max_index))

    for prompt, activation, change in negative_prompts:
        if use_index:
            (
                importances_matrix,
                max_act,
                _,
                tokens_and_activations,
                max_index,
            ) = fast_measure_importance(
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
                max_act,
                _,
                tokens_and_activations,
                max_index,
            ) = fast_measure_importance(
                model,
                layer,
                neuron,
                prompt,
                max_activation=initial_max_act,
                scale_factor=scale_factor,
            )
        info.append((importances_matrix, tokens_and_activations, max_index))

    return info


def train_and_eval(
    model,
    layer,
    neuron,
    aug,
    base_path: str,
    model_name: str,
    activation_matrix,
    layer_ending,
    neuron_store: NeuronStore,
    train_proportion=0.5,
    max_train_size=10,
    max_eval_size=20,
    fire_threshold=0.5,
    random_state=0,
    train_indexes=None,
    return_paths=False,
    **kwargs,
):
    if isinstance(layer, int):
        layer = layer_index_to_name(layer, layer_ending)

    layer_num = int(layer.split(".")[1])
    base_max_act = float(activation_matrix[layer_num, neuron])

    snippets = get_snippets(model_name, layer_num, neuron)

    if train_indexes is None:
        train_snippets, test_snippets = train_test_split(
            snippets, train_size=train_proportion, random_state=random_state
        )
    else:
        train_snippets = [
            snippet for i, snippet in enumerate(snippets) if i in train_indexes
        ]
        test_snippets = [
            snippet for i, snippet in enumerate(snippets) if i not in train_indexes
        ]

    all_train_snippets = train_snippets

    all_info: List[List[Tuple[NDArray[Any], List[List[str | float]], int]]] = []
    for i, snippet in enumerate(all_train_snippets):
        # if i % 10 == 0:
        print(f"Processing {i + 1} of {len(all_train_snippets)}", flush=True)

        pruned_results = fast_prune(
            model, layer, neuron, snippet, return_maxes=True, **kwargs
        )

        for pruned_prompt, _, initial_max_act, truncated_max_act in pruned_results:
            scale_factor = initial_max_act / truncated_max_act

            if pruned_prompt is None:
                continue

            info = augment_and_return(
                model,
                layer,
                neuron,
                aug,
                pruned_prompt,
                base_max_act=base_max_act,
                scale_factor=scale_factor,
                **kwargs,
            )
            all_info.append(info)

    neuron_model = NeuronModel(layer_num, neuron, neuron_store, **kwargs)
    paths = neuron_model.fit(all_info, base_path, model_name)

    print("Fitted model", flush=True)

    max_test_data = []
    for snippet in test_snippets:
        tokens = model.to_tokens(snippet, prepend_bos=True)
        str_tokens = model.to_str_tokens(snippet, prepend_bos=True)
        logits, cache = model.run_with_cache(tokens)
        activations = cache[layer][0, :, neuron].cpu()
        max_test_data.append((str_tokens, activations / base_max_act))

    print("Max Activating Evaluation Data", flush=True)
    try:
        stats = evaluate(
            neuron_model, max_test_data, fire_threshold=fire_threshold, **kwargs
        )
    except Exception as e:
        stats = {}
        print(f"Stats failed with error: {e}", flush=True)

    if return_paths:
        return stats, paths
    return stats
