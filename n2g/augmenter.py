import copy
import typing
from dataclasses import dataclass
from string import punctuation
from typing import Dict, List, Set, Tuple

import numpy as np
import torch
from torch.nn import functional
from transformer_lens.HookedTransformer import HookedTransformer
from transformers import PreTrainedModel, PreTrainedTokenizer  # type: ignore

from .word_tokenizer import WordTokenizer

WordToCasings = Dict[str, List[Tuple[str, int]]]


class Augmenter:
    """Uses BERT to generate variations on input text by masking words and substituting with most likely predictions"""

    def __init__(
        self,
        model: PreTrainedModel,
        model_tokenizer: PreTrainedTokenizer,
        word_tokenizer: WordTokenizer,
        word_to_casings: WordToCasings,
        device: torch.device,
    ):
        self.model: PreTrainedModel = model
        self.model_tokenizer = model_tokenizer
        self.punctuation_set: Set[str] = set(punctuation)
        self.to_strip = " " + punctuation
        self.word_tokenizer = word_tokenizer
        self.device = device
        self.word_to_casings = word_to_casings

    def augment(
        self,
        text: str,
        max_char_position: int,
        important_tokens: Set[str],
        max_augmentations: int,
    ) -> Tuple[List[str], List[int]]:
        joiner = ""
        tokens = self.word_tokenizer(text)

        important_tokens = {token.strip(self.to_strip).lower() for token in important_tokens}

        # Mask important tokens
        masked_token_sets: List[Tuple[List[str], int]] = []
        masked_texts: List[str] = []

        masked_tokens: List[str] = []

        for i, token in enumerate(tokens):
            norm_token = token.strip(self.to_strip).lower() if any(c.isalpha() for c in token) else token

            if not token or self.word_tokenizer.is_split(token) or norm_token not in important_tokens:
                continue

            # If no alphanumeric characters, we'll do a special substitution rather than using BERT
            if not any(c.isalpha() for c in token):
                continue

            before = tokens[:i]
            before_text = joiner.join(before)
            position = len(before_text)

            # Don't bother if we're beyond the max activating token, as these tokens have no effect on the activation
            if position > max_char_position:
                break

            copy_tokens = copy.deepcopy(tokens)
            copy_tokens[i] = "[MASK]"
            masked_token_sets.append((copy_tokens, position))
            masked_texts.append(joiner.join(copy_tokens))

            masked_tokens.append(token)

        assert len(tokens) != 0, "No tokens found in text"
        token = tokens[-1]

        if len(masked_texts) == 0:
            return [], []

        inputs = self.model_tokenizer(masked_texts, padding=True, return_tensors="pt").to(self.device)
        token_probs = functional.softmax(self.model(**inputs).logits, dim=-1).cpu().detach().numpy()
        inputs = inputs.to("cpu")

        new_texts: List[str] = []
        positions: List[int] = []

        for i, (masked_token_set, char_position) in enumerate(masked_token_sets):
            mask_token_id = self.model_tokenizer.mask_token_id  # type: ignore
            mask_token_index = np.argwhere(inputs["input_ids"][i] == mask_token_id)[0, 0]  # type: ignore

            mask_token_probs = token_probs[i, mask_token_index, :]

            # We negate the array before argsort to get the largest, not the smallest, logits
            top_probs = -np.sort(-mask_token_probs).transpose()
            top_tokens = np.argsort(-mask_token_probs).transpose()

            subbed = 0

            # Substitute the given token with the best predictions
            for top_token, top_prob in zip(top_tokens, top_probs):
                if top_prob < 0.00001:
                    break

                candidate_token = self.model_tokenizer.decode(top_token)  # type: ignore

                # Check that the predicted token isn't the same as the token that was already there
                normalised_candidate = (
                    candidate_token.strip(self.to_strip).lower()
                    if candidate_token not in self.punctuation_set
                    else candidate_token
                )
                normalised_token = token.strip(self.to_strip).lower() if token not in self.punctuation_set else token

                if normalised_candidate == normalised_token or not any(c.isalpha() for c in candidate_token):
                    continue

                # Get most common casing of the word
                most_common_casing = self.word_to_casings.get(candidate_token, [(candidate_token, 1)])[0][0]

                original_token = masked_tokens[i]
                # Title case normally has meaning (e.g., start of sentence, in a proper noun, etc.) so follow original
                # token, otherwise use most common
                best_casing = candidate_token.title() if original_token.istitle() else most_common_casing

                new_token_set = copy.deepcopy(masked_token_set)
                # BERT uses ## to denote a tokenisation within a word, so we remove it to glue the word back together
                masked_text = joiner.join(new_token_set)
                new_text = masked_text.replace(self.model_tokenizer.mask_token, best_casing, 1).replace(" ##", "")

                new_texts.append(new_text)
                positions.append(char_position)
                subbed += 1

                if subbed >= max_augmentations:
                    break

        return new_texts, positions


@dataclass
class AugmentationConfig:
    max_length: int = 1024
    inclusion_threshold: float = -0.5
    exclusion_threshold: float = -0.5
    max_augmentations: int = 5


def augment(
    model: HookedTransformer,
    layer: str,
    index: int,
    prompt: str,
    aug: Augmenter,
    important_tokens: Set[str],
    config: AugmentationConfig,
) -> Tuple[List[str], List[str]]:
    """Generate variations of a prompt using an augmenter"""
    prepend_bos = True
    tokens = model.to_tokens(prompt, prepend_bos=prepend_bos)
    str_tokens = typing.cast(
        List[str],
        model.to_str_tokens(prompt, prepend_bos=prepend_bos),  # type: ignore
    )

    if len(tokens[0]) > config.max_length:
        tokens = tokens[0, : config.max_length].unsqueeze(0)

    _logits, cache = model.run_with_cache(tokens)  # type: ignore
    activations = cache[layer][0, :, index]

    initial_max: float = torch.max(activations).cpu().item()
    initial_argmax = typing.cast(int, torch.argmax(activations).cpu().item())
    max_char_position = len("".join(str_tokens[int(prepend_bos) : initial_argmax + 1]))

    positive_prompts: List[str] = [prompt]
    negative_prompts: List[str] = []

    if config.max_augmentations == 0:
        return positive_prompts, negative_prompts

    aug_prompts, aug_positions = aug.augment(
        prompt,
        max_char_position=max_char_position,
        max_augmentations=config.max_augmentations,
        important_tokens=important_tokens,
    )
    if not aug_prompts:
        return positive_prompts, negative_prompts

    aug_tokens = model.to_tokens(aug_prompts, prepend_bos=prepend_bos)

    _aug_logits, aug_cache = model.run_with_cache(aug_tokens)  # type: ignore
    all_aug_activations = aug_cache[layer][:, :, index]

    for aug_prompt, char_position, aug_activations in zip(aug_prompts, aug_positions, all_aug_activations):
        aug_max: float = torch.max(aug_activations).cpu().item()
        aug_argmax = typing.cast(int, torch.argmax(aug_activations).cpu().item())

        # TODO implement this properly - when we mask multiple tokens,
        # if they cross the max_char_position this will not necessarily be correct
        if char_position < max_char_position:
            new_str_tokens: List[str] = model.to_str_tokens(  # type: ignore
                aug_prompt, prepend_bos=prepend_bos
            )
            aug_argmax += len(new_str_tokens) - len(str_tokens)

        proportion_drop: float = (aug_max - initial_max) / initial_max

        if proportion_drop >= config.inclusion_threshold:
            positive_prompts.append(aug_prompt)
        elif proportion_drop < config.exclusion_threshold:
            negative_prompts.append(aug_prompt)

    return positive_prompts, negative_prompts
