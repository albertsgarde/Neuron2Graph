from jaxtyping import Int
from torch import Tensor
from transformer_lens import HookedTransformer  # type: ignore[import]
from transformers import PreTrainedTokenizer, PreTrainedTokenizerBase, PreTrainedTokenizerFast  # type: ignore[import]


class Tokenizer:
    _model: HookedTransformer

    def __init__(self, model: HookedTransformer) -> None:
        self._model = model

    @property
    def tokenizer(self) -> PreTrainedTokenizerBase | PreTrainedTokenizer | PreTrainedTokenizerFast:
        return self._model.tokenizer

    def tokenize(self, samples: list[str], prepend_bos: bool) -> Int[Tensor, "num_samples sample_length"]:
        return self._model.to_tokens(samples, prepend_bos=prepend_bos)

    @property
    def pad_token_id(self) -> int:
        return self._model.tokenizer.pad_token_id

    def str_to_id(self, str_token: str) -> int:
        encoding = self._model.tokenizer.encode(str_token)
        if len(encoding) != 1:
            assert encoding[0] == 15696, (
                f"given string '{str_token}' should be tokenized to exactly one token or to [15696, 19104]. "
                f"Tokenized to '{encoding}'"
            )
            assert encoding[1] == 19104, (
                f"given string '{str_token}' should be tokenized to exactly one token or to [15696, 19104]. "
                f"Tokenized to '{encoding}'"
            )
        return encoding[0]

    def id_to_str(self, token_id: int) -> str:
        return self._model.tokenizer.decode(token_id)

    def tokenize_with_str(self, sample: str, prepend_bos: bool) -> tuple[Int[Tensor, " sample_length"], list[str]]:
        tokens_all, str_tokens_list = self.batch_tokenize_with_str([sample], prepend_bos=prepend_bos)
        assert len(tokens_all) == 1, "tokens_all should have length 1"
        assert len(str_tokens_list) == 1, "str_tokens_list should have length 1"
        tokens: Int[Tensor, " batch_pos"] = tokens_all[0]
        str_tokens: list[str] = str_tokens_list[0]
        return tokens, str_tokens

    def batch_tokenize_with_str(
        self, samples: list[str], prepend_bos: bool
    ) -> tuple[Int[Tensor, "num_samples sample_length"], list[list[str]]]:
        tokens = self._model.to_tokens(samples, prepend_bos=prepend_bos)
        # We need clean_up_tokenization_spaces=False to ensure that the tokenization is reversible.
        str_tokens = [
            self._model.tokenizer.batch_decode(sample, clean_up_tokenization_spaces=False) for sample in tokens
        ]
        return tokens, str_tokens
