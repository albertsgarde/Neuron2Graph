import nltk

nltk.download("stopwords")
from nltk.corpus import stopwords
from string import punctuation
import copy

import scipy.special

import torch.nn.functional as F
import torch
import numpy as np


class FastAugmenter:
    """Uses BERT to generate variations on input text by masking words and substituting with most likely predictions"""

    def __init__(
        self,
        model,
        model_tokenizer,
        word_tokenizer,
        neuron_model,
        word_to_casings,
        device="cuda:0",
    ):
        self.model = model
        self.model_tokenizer = model_tokenizer
        self.stops = set(stopwords.words("english"))
        self.punctuation_set = set(punctuation)
        self.to_strip = " " + punctuation
        self.word_tokenizer = word_tokenizer
        self.device = device
        self.word_to_casings = word_to_casings

    def augment(
        self,
        text,
        max_char_position=None,
        exclude_stopwords=False,
        n=5,
        important_tokens=None,
        **kwargs,
    ):
        joiner = ""
        tokens = self.word_tokenizer(text)

        new_texts = []
        positions = []

        important_tokens = {
            token.strip(self.to_strip).lower() for token in important_tokens
        }

        seen_prompts = set()

        # Gather all tokens to be substituted
        tokens_to_sub = []

        # Mask important tokens
        masked_token_sets = []
        masked_texts = []

        masked_tokens = []

        for i, token in enumerate(tokens):
            norm_token = (
                token.strip(self.to_strip).lower()
                if any(c.isalpha() for c in token)
                else token
            )

            if (
                not token
                or self.word_tokenizer.is_split(token)
                or (exclude_stopwords and norm_token in self.stops)
                or (important_tokens is not None and norm_token not in important_tokens)
            ):
                continue

            # If no alphanumeric characters, we'll do a special substitution rather than using BERT
            if not any(c.isalpha() for c in token):
                continue

            before = tokens[:i]
            before_text = joiner.join(before)
            position = len(before_text)

            # Don't bother if we're beyond the max activating token, as these tokens have no effect on the activation
            if max_char_position is not None and position > max_char_position:
                break

            copy_tokens = copy.deepcopy(tokens)
            copy_tokens[i] = "[MASK]"
            masked_token_sets.append((copy_tokens, position))
            masked_texts.append(joiner.join(copy_tokens))

            masked_tokens.append(token)

        # pprint(masked_texts)
        if len(masked_texts) == 0:
            return [], []

        inputs = self.model_tokenizer(
            masked_texts, padding=True, return_tensors="pt"
        ).to(self.device)
        token_probs = scipy.special.softmax(
            self.model(**inputs).logits.cpu().detach().numpy(), axis=-1
        )
        inputs = inputs.to("cpu")

        chosen_tokens = set()

        new_texts = []
        positions = []

        seen_texts = set()

        for i, (masked_token_set, char_position) in enumerate(masked_token_sets):
            mask_token_index = np.argwhere(
                inputs["input_ids"][i] == self.model_tokenizer.mask_token_id
            )[0, 0]

            mask_token_probs = token_probs[i, mask_token_index, :]

            # We negate the array before argsort to get the largest, not the smallest, logits
            top_probs = -np.sort(-mask_token_probs).transpose()
            top_tokens = np.argsort(-mask_token_probs).transpose()

            subbed = 0

            # Substitute the given token with the best predictions
            for l, (top_token, top_prob) in enumerate(zip(top_tokens, top_probs)):
                if top_prob < 0.00001:
                    break

                candidate_token = self.model_tokenizer.decode(top_token)

                # print(candidate_token, flush=True)

                # Check that the predicted token isn't the same as the token that was already there
                normalised_candidate = (
                    candidate_token.strip(self.to_strip).lower()
                    if candidate_token not in self.punctuation_set
                    else candidate_token
                )
                normalised_token = (
                    token.strip(self.to_strip).lower()
                    if token not in self.punctuation_set
                    else token
                )

                if normalised_candidate == normalised_token or not any(
                    c.isalpha() for c in candidate_token
                ):
                    continue

                # Get most common casing of the word
                most_common_casing = self.word_to_casings.get(
                    candidate_token, [(candidate_token, 1)]
                )[0][0]

                original_token = masked_tokens[i]
                # Title case normally has meaning (e.g., start of sentence, in a proper noun, etc.) so follow original token, otherwise use most common
                best_casing = (
                    candidate_token.title()
                    if original_token.istitle()
                    else most_common_casing
                )

                new_token_set = copy.deepcopy(masked_token_set)
                # BERT uses ## to denote a tokenisation within a word, so we remove it to glue the word back together
                masked_text = joiner.join(new_token_set)
                new_text = masked_text.replace(
                    self.model_tokenizer.mask_token, best_casing, 1
                ).replace(" ##", "")

                if new_text in seen_texts:
                    continue

                new_texts.append(new_text)
                positions.append(char_position)
                subbed += 1

                if subbed >= n:
                    break

        return new_texts, positions


def augment(
    model,
    layer,
    index,
    prompt,
    aug,
    max_length=1024,
    inclusion_threshold=-0.5,
    exclusion_threshold=-0.5,
    n=5,
    **kwargs,
):
    """Generate variations of a prompt using an augmenter"""
    prepend_bos = True
    tokens = model.to_tokens(prompt, prepend_bos=prepend_bos)
    str_tokens = model.to_str_tokens(prompt, prepend_bos=prepend_bos)

    # print(prompt, flush=True)

    if len(tokens[0]) > max_length:
        tokens = tokens[0, :max_length].unsqueeze(0)

    logits, cache = model.run_with_cache(tokens)
    activations = cache[layer][0, :, index]

    initial_max = torch.max(activations).cpu().item()
    initial_argmax = torch.argmax(activations).cpu().item()
    max_char_position = len("".join(str_tokens[int(prepend_bos) : initial_argmax + 1]))

    positive_prompts = [(prompt, initial_max, 1)]
    negative_prompts = []

    if n == 0:
        return positive_prompts, negative_prompts

    aug_prompts, aug_positions = aug.augment(
        prompt, max_char_position=max_char_position, n=n, **kwargs
    )
    if not aug_prompts:
        return positive_prompts, negative_prompts

    aug_tokens = model.to_tokens(aug_prompts, prepend_bos=prepend_bos)

    aug_logits, aug_cache = model.run_with_cache(aug_tokens)
    all_aug_activations = aug_cache[layer][:, :, index]

    for aug_prompt, char_position, aug_activations in zip(
        aug_prompts, aug_positions, all_aug_activations
    ):
        aug_max = torch.max(aug_activations).cpu().item()
        aug_argmax = torch.argmax(aug_activations).cpu().item()

        # TODO implement this properly - when we mask multiple tokens, if they cross the max_char_position this will not necessarily be correct
        if char_position < max_char_position:
            new_str_tokens = model.to_str_tokens(aug_prompt, prepend_bos=prepend_bos)
            aug_argmax += len(new_str_tokens) - len(str_tokens)

        proportion_drop = (aug_max - initial_max) / initial_max

        if proportion_drop >= inclusion_threshold:
            positive_prompts.append((aug_prompt, aug_max, proportion_drop))
        elif proportion_drop < exclusion_threshold:
            negative_prompts.append((aug_prompt, aug_max, proportion_drop))

    return positive_prompts, negative_prompts
