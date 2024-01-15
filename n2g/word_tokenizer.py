import re
from collections import defaultdict
from typing import Dict, List, Set, Tuple


class WordTokenizer:
    """Simple tokenizer for splitting text into words"""

    def __init__(self, split_tokens: Set[str], stick_tokens: Set[str]):
        self.split_tokens = split_tokens
        self.stick_tokens = stick_tokens

    def __call__(self, text: str) -> List[str]:
        return self.tokenize(text)

    def is_split(self, char: str) -> bool:
        """Split on any non-alphabet chars unless excluded, and split on any specified chars"""
        return char in self.split_tokens or (
            not char.isalpha() and char not in self.stick_tokens
        )

    def tokenize(self, text: str) -> List[str]:
        """Tokenize text, preserving all characters"""
        tokens: List[str] = []
        current_token = ""
        for char in text:
            if self.is_split(char):
                tokens.append(current_token)
                tokens.append(char)
                current_token = ""
                continue
            current_token += char
        tokens.append(current_token)
        tokens = [token for token in tokens if token]
        return tokens


splitter = re.compile("[\\.!\\n]")  # ignore: type


def sentence_tokenizer(
    str_tokens: List[str],
) -> Tuple[List[List[str]], Dict[int, List[int]], Dict[int, int]]:
    """Split tokenized text into sentences"""
    sentences: List[List[str]] = []
    sentence: List[str] = []
    sentence_to_token_indices: Dict[int, List[int]] = defaultdict(list)
    token_to_sentence_indices: Dict[int, int] = {}

    for i, str_token in enumerate(str_tokens):
        sentence.append(str_token)
        sentence_to_token_indices[len(sentences)].append(i)
        token_to_sentence_indices[i] = len(sentences)
        if splitter.search(str_token) is not None or i + 1 == len(str_tokens):
            sentences.append(sentence)
            sentence = []

    return sentences, sentence_to_token_indices, token_to_sentence_indices
