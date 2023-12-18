from pprint import pprint
import re
import math
from collections import defaultdict
import json
from typing import List, Tuple
import os
import random
import time
import sys
import traceback

import numpy as np

import requests

import torch
from transformers import AutoModelForMaskedLM
from transformers import AutoTokenizer
from transformer_lens import HookedTransformer

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from n2g.neuron_model import NeuronModel

from .n2g import FastAugmenter, WordTokenizer, NeuronStore
import n2g

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


def layer_index_to_name(layer_index):
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
    full_initial_argmax = torch.argmax(activations).cpu().item()

    (
        sentences,
        sentence_to_token_indices,
        token_to_sentence_indices,
    ) = n2g.sentence_tokenizer(str_tokens)

    # print(activation_threshold * full_initial_max, flush=True)

    strong_indices = torch.where(
        activations >= token_activation_threshold * full_initial_max
    )[0]
    strong_activations = activations[strong_indices].cpu()
    strong_indices = strong_indices.cpu()

    # print(strong_activations, flush=True)
    # print(strong_indices, flush=True)

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
        # print(strong_sentence_index, initial_argmax, initial_max, flush=True)

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

            # print(count, len(full_prior), flush=True)

            if (
                not count == len(full_prior)
                and count >= skip_threshold
                and count % skip_interval != 0
            ):
                continue

            # print("Made it!", flush=True)

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
            # print("length", len(truncated_batch), flush=True)
            # pprint(truncated_batch)

            truncated_tokens = model.to_tokens(truncated_batch, prepend_bos=prepend_bos)

            # pprint(truncated_tokens)

            logits, cache = model.run_with_cache(truncated_tokens)
            all_truncated_activations = cache[layer][:, :, neuron].cpu()

            # print("shape", all_truncated_activations.shape, flush=True)

            for j, truncated_activations in enumerate(all_truncated_activations):
                num_added_tokens = added_tokens_batch[j]
                # print("single shape", truncated_activations.shape, flush=True)
                truncated_argmax = (
                    torch.argmax(truncated_activations).cpu().item() + num_added_tokens
                )
                final_max_index = torch.argmax(truncated_activations).cpu().item()

                if prepend_bos:
                    truncated_argmax -= 1
                    final_max_index -= 1
                truncated_max = torch.max(truncated_activations).cpu().item()

                # trunc_logits, trunc_cache = model.run_with_cache(model.to_tokens(truncated_batch[j], prepend_bos=prepend_bos))
                # trunc_activations = trunc_cache[layer][0, :, neuron]

                # print(truncated_activations, flush=True)
                # print(trunc_activations, flush=True)
                # print("truncated_argmax", truncated_argmax, flush=True)
                # print(truncated_max, flush=True)

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

        # if shortest_successful_prompt is None:
        #   pruned_sentence = "".join(relevant_str_tokens)
        #   final_max_index = initial_argmax
        # else:
        pruned_sentence = "".join(
            shortest_successful_prompt
        )  # if shortest_successful_prompt is not None else shortest_prompt

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
    layer,
    neuron,
    prompt,
    initial_argmax=None,
    max_length=1024,
    max_activation=None,
    masking_token=1,
    threshold=0.8,
    scale_factor=1,
    return_all=False,
    activation_threshold=0.1,
    **kwargs,
):
    """Compute a measure of token importance by masking each token and measuring the drop in activation on the max activating token"""

    prepend_bos = True
    tokens = model.to_tokens(prompt, prepend_bos=prepend_bos)
    str_tokens = model.to_str_tokens(prompt, prepend_bos=prepend_bos)

    if len(tokens[0]) > max_length:
        tokens = tokens[0, :max_length].unsqueeze(0)

    # logits, cache = model.run_with_cache(tokens)

    # print(tokens_and_activations, flush=True)

    importances_matrix = []

    shortest_successful_prompt = None
    # cutoff = 50

    masked_prompts = tokens.repeat(len(tokens[0]) + 1, 1)

    # print(f"{len(masked_prompts)=}, {initial_argmax=}, {starting_point=}", flush=True)

    for i in range(1, len(masked_prompts)):
        masked_prompts[i, i - 1] = masking_token

    # for i, str_token in enumerate(str_tokens):
    #   if i >= cutoff:
    #     break

    #   masked_tokens = tokens

    #   if i >= len(masked_tokens[0]):
    #     continue

    #   token_to_mask = copy.deepcopy(tokens[0, i])
    #   masked_tokens[0, i] = masking_token

    #   masked_prompts.append(masked_tokens[0])
    #   tokens[0, i] = token_to_mask

    # pprint(masked_prompts)

    logits, cache = model.run_with_cache(masked_prompts)
    all_masked_activations = cache[layer][1:, :, neuron].cpu()

    activations = cache[layer][0, :, neuron].cpu()

    if initial_argmax is None:
        initial_argmax = torch.argmax(activations).cpu().item()
    else:
        # This could be wrong
        initial_argmax = min(initial_argmax, len(activations) - 1)

    # print(activations, flush=True)
    # print(activation_threshold, flush=True)
    # activation_indexes = [i for i, activation in enumerate(activations) if activation * scale_factor / max_activation > activation_threshold]
    # print(activation_indexes, flush=True)
    # final_activating = initial_argmax if len(activation_indexes) == 0 else activation_indexes[-1]

    initial_max = activations[initial_argmax].cpu().item()

    if max_activation is None:
        max_activation = initial_max
    scale = min(1, initial_max / max_activation)

    # print("scale_factor measure_importance", scale_factor, flush=True)

    tokens_and_activations = [
        [str_token, round(activation.cpu().item() * scale_factor / max_activation, 3)]
        for str_token, activation in zip(str_tokens, activations)
    ]
    important_tokens = []
    tokens_and_importances = [[str_token, 0] for str_token in str_tokens]

    for i, masked_activations in enumerate(all_masked_activations):
        if return_all:
            # Get importance of the given token for all tokens
            importances_row = []
            for j, activation in enumerate(masked_activations):
                activation = activation.cpu().item()
                normalised_activation = 1 - (activation / activations[j].cpu().item())
                importances_row.append((str_tokens[j], normalised_activation))

            # for j, str_token in enumerate(str_tokens[cutoff:]):
            #   importances_row.append((str_token, 0))

            # print("importances_row", importances_row, flush=True)
            importances_matrix.append(np.array(importances_row))

        masked_max = masked_activations[initial_argmax].cpu().item()
        normalised_activation = 1 - (masked_max / initial_max)

        str_token = tokens_and_importances[i][0]
        tokens_and_importances[i][1] = normalised_activation
        if normalised_activation >= threshold and str_token != "<|endoftext|>":
            important_tokens.append(str_token)

    # for i, str_token in enumerate(str_tokens[cutoff:]):
    #   tokens_and_importances.append((str_token, 0))

    if return_all:
        # Flip so we have the importance of all tokens for a given token
        importances_matrix = np.array(importances_matrix)
        return (
            importances_matrix,
            initial_max,
            important_tokens,
            tokens_and_activations,
            initial_argmax,
        )

    return (
        tokens_and_importances,
        initial_max,
        important_tokens,
        tokens_and_activations,
        initial_argmax,
    )


def train_and_eval(
    model,
    layer,
    neuron,
    aug,
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
        layer = layer_index_to_name(layer)

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
    # train_data, test_data = train_test_split(data, train_size=train_proportion, random_state=0)

    # train_data_snippets = ["".join(tokens) for tokens, activations in train_data if any(activation > fire_threshold for activation in activations)][:max_train_size]
    train_data_snippets = []
    all_train_snippets = train_snippets + train_data_snippets

    all_info = []
    for i, snippet in enumerate(all_train_snippets):
        # if i % 10 == 0:
        print(f"Processing {i + 1} of {len(all_train_snippets)}", flush=True)

        pruned_results = fast_prune(
            model, layer, neuron, snippet, return_maxes=True, **kwargs
        )

        for pruned_prompt, _, initial_max_act, truncated_max_act in pruned_results:
            # tokens = model.to_tokens(pruned_prompt, prepend_bos=True)
            # str_tokens = model.to_str_tokens(pruned_prompt, prepend_bos=True)
            # logits, cache = model.run_with_cache(tokens)
            # activations = cache[layer][0, :, neuron].cpu()
            # max_pruned_activation = torch.max(activations).item()
            scale_factor = initial_max_act / truncated_max_act
            # scale_factor = 1

            # print(scale_factor, flush=True)
            # scaled_activations = activations * scale_factor / base_max_act

            # print(list(zip(str_tokens, activations)), flush=True)

            # print(pruned_prompt, flush=True)

            # print(len(pruned_prompt), flush=True)

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

    neuron_model = NeuronModel(layer_num, neuron)
    paths = neuron_model.fit(all_info, base_path, model_name)

    print("Fitted model", flush=True)

    max_test_data = []
    for snippet in test_snippets:
        # pruned_prompt, _ = prune(model, layer, neuron, snippet, **kwargs)
        # if pruned_prompt is None:
        #   continue
        tokens = model.to_tokens(snippet, prepend_bos=True)
        str_tokens = model.to_str_tokens(snippet, prepend_bos=True)
        logits, cache = model.run_with_cache(tokens)
        activations = cache[layer][0, :, neuron].cpu()
        max_test_data.append((str_tokens, activations / base_max_act))

    # pprint(max_test_data[0])
    # print("\n\n", flush=True)
    # pprint(test_data[0])

    # print("Evaluation data", flush=True)
    # test_data = test_data[:max_eval_size]
    # evaluate(neuron_model, test_data, fire_threshold=fire_threshold, **kwargs)

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
):
    info = []
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
        return_all=True,
    )

    if base_max_act is not None:
        initial_max_act = base_max_act

    positive_prompts, negative_prompts = n2g.augment(
        model,
        layer,
        neuron,
        pruned_prompt,
        aug,
        important_tokens=set(important_tokens),
        **kwargs,
    )

    for i, (prompt, activation, change) in enumerate(positive_prompts):
        title = prompt
        if i == 0:
            title = "Original - " + prompt

        #   print("Original", flush=True)
        #   print(prompt, "\n", flush=True)
        # elif i > 1:
        #   print("Augmented", flush=True)
        #   print(prompt, "\n", flush=True)

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
                return_all=True,
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
                return_all=True,
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
                return_all=True,
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
                return_all=True,
            )
        info.append((importances_matrix, tokens_and_activations, max_index))

    return info


def evaluate(neuron_model, data, fire_threshold=0.5, **kwargs):
    y = []
    y_pred = []
    y_act = []
    y_pred_act = []
    for prompt_tokens, activations in data:
        # print("truth", flush=True)
        non_zero_indices = [
            i for i, activation in enumerate(activations) if activation > 0
        ]
        start = max(0, non_zero_indices[0] - 10)
        end = min(len(prompt_tokens) - 1, non_zero_indices[-1] + 10)
        pred_activations = neuron_model.forward(
            [prompt_tokens], return_activations=True
        )[0]

        y_act.extend(activations)
        y_pred_act.extend(pred_activations)

        important_context = list(zip(prompt_tokens, activations, pred_activations))[
            start:end
        ]

        # print(important_context, flush=True)
        # print(len(pred_activations), flush=True)
        pred_firings = [
            int(pred_activation >= fire_threshold)
            for pred_activation in pred_activations
        ]
        firings = [int(activation >= fire_threshold) for activation in activations]
        y_pred.extend(pred_firings)
        y.extend(firings)
    # print(len(y), len(y_pred), flush=True)
    print(classification_report(y, y_pred), flush=True)
    report = classification_report(y, y_pred, output_dict=True)

    y_act = np.array(y_act)
    y_pred_act = np.array(y_pred_act)

    # y_pred_act = y_pred_act[y_act > 0.5]
    # y_act = y_act[y_act > 0.5]

    # print(y_act[:10], flush=True)
    # print(y_pred_act[:10], flush=True)

    # y_pred_act = y_pred_act * np.mean(y_act) / np.mean(y_pred_act)
    # y_pred_act =

    act_diff = y_pred_act - y_act
    mse = np.mean(np.power(act_diff, 2))
    variance = np.var(y_act)
    correlation = 1 - (mse / variance)
    # print(f"{correlation=:.3f}, {mse=:.3f}, {variance=:.4f}", flush=True)

    report["correlation"] = correlation
    return report


def get_summary_stats(path, verbose=True):
    summary_stats = []
    summary_stds = []

    with open(path) as ifh:
        stats = json.load(ifh)

    missing = 0

    random.seed(0)

    inelegible_count = 0

    precision_case = 0

    for layer, layer_stats in stats.items():
        # pprint(layer_stats)
        eligible_neurons = [
            neuron
            for neuron, neuron_stats in layer_stats.items()
            if "1" in neuron_stats
        ]
        # neuron_sample = set(random.sample(eligible_neurons, 50))
        eligible_neurons = set(eligible_neurons)

        aggr_stats_dict = {
            "Inactivating": defaultdict(list),
            "Activating": defaultdict(list),
        }
        for neuron, neuron_stats in layer_stats.items():
            if neuron not in eligible_neurons:
                inelegible_count += 1
                continue

            aggr_stats_dict["Inactivating"]["Precision"].append(
                neuron_stats["0"]["precision"]
            )
            aggr_stats_dict["Inactivating"]["Recall"].append(
                neuron_stats["0"]["recall"]
            )
            aggr_stats_dict["Inactivating"]["F1"].append(neuron_stats["0"]["f1-score"])

            # print(neuron_stats["0"]["precision"], neuron_stats["0"]["recall"], neuron_stats["0"]["f1-score"],
            #       neuron_stats["1"]["precision"], neuron_stats["1"]["recall"], neuron_stats["1"]["f1-score"])

            # If we didn't predict anything as activating, treat this as 100% precision rather than 0%
            if neuron_stats["0"]["recall"] == 1 and neuron_stats["1"]["recall"] == 0:
                # print("Precision case", flush=True)
                precision_case += 1
                neuron_stats["1"]["precision"] = 1.0

            aggr_stats_dict["Activating"]["Precision"].append(
                neuron_stats["1"]["precision"]
            )
            aggr_stats_dict["Activating"]["Recall"].append(neuron_stats["1"]["recall"])
            aggr_stats_dict["Activating"]["F1"].append(neuron_stats["1"]["f1-score"])

        #   if neuron == "20":
        #     break
        # break

        # if neuron_stats["1"]["recall"] > 0.8:
        #   print(f'{layer}, {neuron}, {neuron_stats["1"]["precision"]:.3f}, {neuron_stats["1"]["recall"]:.3f}, {neuron_stats["1"]["f1-score"]:.3f}', flush=True)
        if verbose:
            print(
                "Neurons Evaluated:", len(aggr_stats_dict["Inactivating"]["Precision"])
            )

        avg_stats_dict = {"Inactivating": {}, "Activating": {}}
        std_stats_dict = {"Inactivating": {}, "Activating": {}}
        for token_type, inner_stats_dict in aggr_stats_dict.items():
            for stat_type, stat_arr in inner_stats_dict.items():
                avg_stats_dict[token_type][stat_type] = round(np.mean(stat_arr), 3)
                std_stats_dict[token_type][stat_type] = round(np.std(stat_arr), 3)

        summary_stats.append(avg_stats_dict)
        summary_stds.append(std_stats_dict)
        # break

    if verbose:
        for layer, (summary, std_summary) in enumerate(
            zip(summary_stats, summary_stds)
        ):
            print("\n", flush=True)
            pprint(summary)
            pprint(std_summary)

        print(f"{inelegible_count=}", flush=True)
        print(f"{precision_case=}", flush=True)

    return summary_stats


def run_training(
    layers, neurons, folder_name, sample_num=None, params=None, start_neuron=None
):
    if params is None or not params:
        params = {
            "importance_threshold": 0.75,
            "n": 5,
            "max_train_size": None,
            "train_proportion": 0.5,
            "max_eval_size": 0.5,
            "activation_threshold": 0.5,
            "token_activation_threshold": 1,
            "fire_threshold": 0.5,
        }
    print(f"{params=}\n", flush=True)
    random.seed(0)

    all_neuron_indices = [i for i in range(neurons)]

    if not os.path.exists(f"{base_path}/neuron_graphs/{model_name}"):
        os.mkdir(f"{base_path}/neuron_graphs/{model_name}")

    neuron_store = NeuronStore(
        f"{base_path}/neuron_graphs/{model_name}/neuron_store.json"
    )

    folder_path = os.path.join(base_path, f"neuron_graphs/{model_name}/{folder_name}")

    if not os.path.exists(folder_path):
        print("Making", folder_path, flush=True)
        os.mkdir(folder_path)

    if os.path.exists(f"{folder_path}/stats.json"):
        with open(f"{folder_path}/stats.json") as ifh:
            all_stats = json.load(ifh)

    else:
        all_stats = {}

    printerval = 10

    for layer in layers:
        t1 = time.perf_counter()

        if sample_num is None:
            chosen_neuron_indices = all_neuron_indices
        else:
            chosen_neuron_indices = random.sample(all_neuron_indices, sample_num)
            chosen_neuron_indices = sorted(chosen_neuron_indices)

        all_stats[layer] = {}
        for i, neuron in enumerate(chosen_neuron_indices):
            if start_neuron is not None and neuron < start_neuron:
                continue

            print(f"{layer=} {neuron=}", flush=True)
            try:
                stats = train_and_eval(
                    model,
                    layer,
                    neuron,
                    fast_aug,
                    folder_name=folder_name,
                    neuron_store=neuron_store,
                    **params,
                )

                all_stats[layer][neuron] = stats

                if i % printerval == 0:
                    t2 = time.perf_counter()
                    elapsed = t2 - t1
                    rate = printerval / elapsed
                    remaining = (len(chosen_neuron_indices) - i) / rate / 60
                    print(
                        f"{i} complete, batch took {elapsed / 60:.2f} mins, {rate=:.2f} neurons/s, {remaining=:.1f} mins"
                    )
                    t1 = t2

                    neuron_store.save()
                    with open(f"{folder_path}/stats.json", "w") as ofh:
                        json.dump(all_stats, ofh, indent=2)

            except Exception as e:
                print(traceback.format_exc(), flush=True)
                print("Failed", flush=True)

    neuron_store.save()
    with open(f"{folder_path}/stats.json", "w") as ofh:
        json.dump(all_stats, ofh, indent=2)


def cmd_arguments() -> Tuple[str, str, List[int], int]:
    """
    Gets model name, layer ending, layers, and neurons from the command line arguments if available.
    Layer ending should be either "mid" for SoLU models or "post" for GeLU models.
    The "mlp.hook_" prefix is added automatically.
    Layers are given either as `start:end` or as a comma separated list of layer indices.
    """
    args = sys.argv[1:]
    num_arguments = len(args)
    if num_arguments < 4:
        raise Exception("Not enough arguments")
    model_name = args[0]
    layer_ending = f"mlp.hook_{args[1]}"
    layers_arg = args[2]
    if ":" in layers_arg:
        layer_range = layers_arg.split(":")
        layers = range(int(layer_range[0]), int(layer_range[1]))
    else:
        layers = [int(layer_index_str) for layer_index_str in layers_arg.split(",")]
    neurons_per_layer = int(args[3])

    return model_name, layer_ending, layers, neurons_per_layer


if __name__ == "__main__":
    """
    Instructions:
    Download word_to_casings.json from the repo and put in data/
    Set layer_ending to "mlp.hook_mid" for SoLU models and "mlp.hook_post" for GeLU models
    Download from the repo or scrape (with scrape.py) the activation matrix for the model and put in data/
    Set model_name to the name of the model you want to run for
    Set the parameters in the run section as desired
    Run this file!

    It will create neuron graphs for the specified layers and neurons in neuron_graphs/model_name/folder_name
    It'll also save the stats for each neuron in neuron_graphs/model_name/folder_name/stats.json
    And it will save the neuron store in neuron_graphs/model_name/neuron_store.json
    """

    model_name, layer_ending, layers, neurons_per_layer = cmd_arguments()
    # Uncomment and overwrite if you would rather specify the model name and layer ending here.
    # model_name = "gpt2-small"
    # layer_ending = "mlp.hook_post"

    print(
        f"Running N2G for model {model_name} layers {layers}. Using layer ending {layer_ending} and {neurons_per_layer} neurons per layer."
    )

    # ================ Setup ================
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}", flush=True)
    model = HookedTransformer.from_pretrained(model_name).to(device)

    base_path = os.path.abspath(
        os.path.join(os.path.dirname(os.path.abspath(__file__)))
    )

    # Save the activation matrix for the model to data/
    activation_matrix_path = os.path.join(
        base_path, f"data/activation_matrix-{model_name}.json"
    )
    if not os.path.exists(activation_matrix_path):
        raise Exception(
            f"Activation matrix not found for model {model_name}. Either download it from the repo or scrape it with `scrape.py`."
        )
    with open(activation_matrix_path) as ifh:
        activation_matrix = json.load(ifh)
        activation_matrix = np.array(activation_matrix)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}", flush=True)
    aug_model_checkpoint = "distilbert-base-uncased"
    aug_model = AutoModelForMaskedLM.from_pretrained(aug_model_checkpoint).to(device)
    aug_tokenizer = AutoTokenizer.from_pretrained(aug_model_checkpoint)

    with open(f"{base_path}/data/word_to_casings.json", encoding="utf-8") as ifh:
        word_to_casings = json.load(ifh)

    stick_tokens = {"'"}
    word_tokenizer = WordTokenizer(set(), stick_tokens)
    fast_aug = FastAugmenter(
        aug_model, aug_tokenizer, word_tokenizer, model, word_to_casings
    )

    # main()

    if not os.path.exists(f"{base_path}/neuron_graphs"):
        os.mkdir(f"{base_path}/neuron_graphs")

    # ================ Run ================
    # Run training for the specified layers and neurons

    folder_name = "layer_0"

    # Override params as desired - sensible defaults are set in run_training
    params = {}

    run_training(
        # List of layers to run for
        layers=layers,
        # Number of neurons in each layer
        neurons=neurons_per_layer,
        # Neuron to start at (useful for resuming - None to start at 0)
        start_neuron=None,
        # Folder to save results in
        folder_name=folder_name,
        # Number of neurons to sample from each layer (None for all neurons)
        sample_num=None,
        params=params,
    )

    get_summary_stats(
        f"{base_path}/neuron_graphs/{model_name}/{folder_name}/stats.json"
    )
