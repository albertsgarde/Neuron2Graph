import math
import typing
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple, TypeVar

import numpy as np
import torch
from jaxtyping import Float, Int
from numpy.typing import NDArray
from torch import Tensor

from . import augmenter, neuron_model as fit_model, word_tokenizer
from .augmenter import AugmentationConfig, Augmenter
from .feature_model import FeatureModel
from .neuron_model import Sample
from .tokenizer import Tokenizer

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


@dataclass
class PruneConfig:
    max_length: int = 1024
    proportion_threshold: float = -0.5
    absolute_threshold: Optional[float] = None
    token_activation_threshold: float = 1
    window: int = 0
    cutoff: int = 30
    batch_size: int = 4
    max_post_context_tokens: int = 5
    skip_threshold: int = 0
    skip_interval: int = 5
    prepend_bos: bool = True


def prune(
    feature_activation: Callable[
        [Int[Tensor, "num_samples sample_length"]], Float[Tensor, "num_samples sample_length"]
    ],
    tokenizer: Tokenizer,
    prompt: str,
    config: PruneConfig,
) -> List[Tuple[str, float, float]]:
    """
    Prune an input prompt to the shortest string that preserves x% of neuron activation on the most activating token.
    """

    tokens: Int[Tensor, " sample_length"]
    tokens, str_tokens = tokenizer.tokenize_with_str(prompt, prepend_bos=config.prepend_bos)

    # If we instead took pre-tokenized input, we would not need this check.
    if tokens.shape[-1] > config.max_length:
        tokens = tokens[: config.max_length]

    activations: Float[Tensor, " sample_length"] = feature_activation(tokens)[:].cpu()
    assert activations.shape == tokens.shape

    full_initial_max = torch.max(activations).cpu().item()
    if full_initial_max <= 0:
        return []

    (
        sentences,
        token_to_sentence_indices,
    ) = word_tokenizer.sentence_tokenizer(str_tokens)

    strong_indices: Tensor = torch.where(  # type: ignore
        activations >= config.token_activation_threshold * full_initial_max
    )[0]
    strong_activations = activations[strong_indices].cpu()
    strong_indices = strong_indices.cpu()

    pruned_sentences: List[str] = []
    initial_maxes: List[float] = []
    truncated_maxes: List[float] = []

    # For each strong activation, find the shortest prompt that preserves the activation
    for initial_argmax_tensor, initial_max_tensor in zip(strong_indices, strong_activations):
        initial_argmax: int = initial_argmax_tensor.item()  # type: ignore
        initial_max: float = initial_max_tensor.item()

        max_sentence_index = token_to_sentence_indices[initial_argmax]
        relevant_str_tokens = [str_token for sentence in sentences[: max_sentence_index + 1] for str_token in sentence]

        prior_context = relevant_str_tokens[: initial_argmax + 1]

        post_context = relevant_str_tokens[initial_argmax + 1 :]

        shortest_successful_prompt: Optional[str] = None

        truncated_prompts: List[str] = []
        added_tokens: List[int] = []

        count = 0
        full_prior = prior_context[: max(0, initial_argmax - config.window + 1)]

        for i in reversed(range(len(full_prior))):
            count += 1

            if count > config.cutoff:
                break

            if not count == len(full_prior) and count >= config.skip_threshold and count % config.skip_interval != 0:
                continue

            truncated_prompt = prior_context[i:]
            joined = "".join(truncated_prompt)
            truncated_prompts.append(joined)
            added_tokens.append(i)

        batched_truncated_prompts = batch(truncated_prompts, config.batch_size)
        batched_added_tokens = batch(added_tokens, config.batch_size)

        finished = False

        truncated_max: Optional[float] = None
        shortest_successful_prompt: str | None = None
        for i, (truncated_batch, added_tokens_batch) in enumerate(zip(batched_truncated_prompts, batched_added_tokens)):
            truncated_tokens = tokenizer.tokenize(truncated_batch, prepend_bos=config.prepend_bos)

            all_truncated_activations = feature_activation(truncated_tokens).cpu()

            for j, truncated_activations in enumerate(all_truncated_activations):
                num_added_tokens = added_tokens_batch[j]
                truncated_argmax = torch.argmax(truncated_activations).cpu().item() + num_added_tokens

                if config.prepend_bos:
                    truncated_argmax -= 1
                truncated_max = torch.max(truncated_activations).cpu().item()

                shortest_prompt = truncated_batch[j]

                if (
                    truncated_argmax == initial_argmax
                    and (
                        (truncated_max - initial_max) / initial_max > config.proportion_threshold
                        or (config.absolute_threshold is not None and truncated_max >= config.absolute_threshold)
                    )
                ) or (i == len(batched_truncated_prompts) - 1 and j == len(all_truncated_activations) - 1):
                    if truncated_max > 0:
                        shortest_successful_prompt = shortest_prompt
                    finished = True
                    break

            if finished:
                break

        if shortest_successful_prompt is not None:
            pruned_sentence: str = shortest_successful_prompt

            pruned_sentence += "".join(post_context[: config.max_post_context_tokens])

            pruned_sentences.append(pruned_sentence)
            initial_maxes.append(initial_max)
            assert truncated_max is not None, "Truncated max is None"
            truncated_maxes.append(truncated_max)

    return list(zip(pruned_sentences, initial_maxes, truncated_maxes))


@dataclass
class ImportanceConfig:
    max_length: int = 1024
    masking_token: int = 1
    threshold: float = 0.8
    prepend_bos: bool = True


def measure_importance(
    feature_activation: Callable[
        [Int[Tensor, "num_samples sample_length"]], Float[Tensor, "num_samples sample_length"]
    ],
    tokenizer: Tokenizer,
    prompt: str,
    max_activation: float,
    scale_factor: float,
    config: ImportanceConfig,
) -> Tuple[NDArray[np.float32], List[str], List[Tuple[str, float]]]:
    """Compute a measure of token importance by masking each token and measuring the drop in activation on
    the max activating token"""

    tokens: Int[Tensor, "1 prompt_length"]
    tokens, str_tokens_list = tokenizer.batch_tokenize_with_str([prompt], prepend_bos=config.prepend_bos)
    str_tokens: list[str] = str_tokens_list[0]

    for str_token in str_tokens:
        assert tokenizer.str_to_id(str_token) is not None

    if len(tokens[0]) > config.max_length:
        tokens = tokens[0, : config.max_length].unsqueeze(0)

    masked_prompts: Int[Tensor, "prompt_length+1 prompt_length"] = tokens.repeat(len(tokens[0]) + 1, 1)

    for i in range(1, len(masked_prompts)):
        masked_prompts[i, i - 1] = config.masking_token

    all_activations: Float[Tensor, "prompt_length+1 prompt_length"] = feature_activation(masked_prompts).cpu()

    all_masked_activations: Float[Tensor, "prompt_length prompt_length"] = all_activations[1:, :]
    unmasked_activations: Float[Tensor, " prompt_length"] = all_activations[0, :]

    initial_argmax = typing.cast(int, torch.argmax(unmasked_activations).item())

    initial_max: float = typing.cast(float, unmasked_activations[initial_argmax].item())

    tokens_and_activations: List[Tuple[str, float]] = [
        (str_token, round(activation.item() * scale_factor / max_activation, 3))
        for str_token, activation in zip(str_tokens, unmasked_activations)
    ]
    important_tokens: List[str] = []
    tokens_and_importances: List[Tuple[str, float]] = [(str_token, 0) for str_token in str_tokens]

    importances_matrix: Float[Tensor, "prompt_length prompt_length"] = torch.where(
        unmasked_activations == 0,
        torch.zeros_like(all_masked_activations),
        1 - all_masked_activations / unmasked_activations,
    )

    masked_maxes: Int[Tensor, " prompt_length"] = all_masked_activations[:, initial_argmax]
    token_importances = 1 - (masked_maxes / initial_max) if initial_max != 0 else torch.zeros_like(masked_maxes)
    tokens_and_importances = [
        (str_token, importance.item()) for str_token, importance in zip(str_tokens, token_importances, strict=True)
    ]
    important_tokens = [
        str_token
        for str_token, importance in tokens_and_importances
        if importance >= config.threshold and str_token != "<|endoftext|>"
    ]

    return (
        np.array(importances_matrix),
        important_tokens,
        tokens_and_activations,
    )


def augment_and_return(
    feature_activation: Callable[
        [Int[Tensor, "num_samples sample_length"]], Float[Tensor, "num_samples sample_length"]
    ],
    tokenizer: Tokenizer,
    aug: Augmenter,
    pruned_prompt: str,
    base_max_act: float,
    scale_factor: float,
    importance_config: ImportanceConfig,
    augmentation_config: AugmentationConfig,
) -> list[Sample]:
    samples: list[Sample] = []
    (
        _importances_matrix,
        important_tokens,
        _tokens_and_activations,
    ) = measure_importance(
        feature_activation,
        tokenizer,
        pruned_prompt,
        max_activation=base_max_act,
        scale_factor=scale_factor,
        config=importance_config,
    )

    prompts = augmenter.augment(
        feature_activation,
        tokenizer,
        pruned_prompt,
        aug,
        important_tokens=set(important_tokens),
        config=augmentation_config,
    )

    for prompt in prompts:
        (
            importances_matrix,
            _important_tokens,
            tokens_and_activations,
        ) = measure_importance(
            feature_activation,
            tokenizer,
            prompt,
            max_activation=base_max_act,
            scale_factor=scale_factor,
            config=importance_config,
        )
        samples.append(Sample(importances_matrix, tokens_and_activations))

    return samples


@dataclass
class FitConfig:
    prune_config: PruneConfig
    importance_config: ImportanceConfig
    augmentation_config: AugmentationConfig
    activation_threshold: float = 0.5
    importance_threshold: float = 0.75


def fit_neuron_model(
    feature_activation: Callable[
        [Int[Tensor, "num_samples sample_length"]], Float[Tensor, "num_samples sample_length"]
    ],
    tokenizer: Tokenizer,
    train_samples: list[str],
    augmenter: Augmenter,
    base_max_act: float,
    config: FitConfig,
) -> FeatureModel:
    all_samples: list[list[Sample]] = []
    for i, sample in enumerate(train_samples):
        print(f"Processing {i + 1} of {len(train_samples)}", flush=True)

        pruned_results = prune(feature_activation, tokenizer, sample, config=config.prune_config)

        for pruned_prompt, initial_max_act, truncated_max_act in pruned_results:
            scale_factor = initial_max_act / truncated_max_act

            samples = augment_and_return(
                feature_activation,
                tokenizer,
                augmenter,
                pruned_prompt,
                base_max_act=base_max_act,
                scale_factor=scale_factor,
                importance_config=config.importance_config,
                augmentation_config=config.augmentation_config,
            )
            all_samples.append(samples)

    feature_model = FeatureModel.from_samples(
        tokenizer, all_samples, config.importance_threshold, config.activation_threshold
    )

    return feature_model
