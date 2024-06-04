from dataclasses import dataclass

from n2g_rs import (
    FeatureModel as RustFeatureModel,
    FeatureModelNode as RustFeatureModelNode,
    Pattern,
    PatternToken,
    Token,
)

from n2g import neuron_model
from n2g.neuron_model import Element, Line, NeuronModel, NeuronNode, Sample
from n2g.tokenizer import Tokenizer


def _element_to_pattern_token(tokenizer: Tokenizer, element: Element) -> tuple[PatternToken, float] | None:
    if element.is_end:
        return None
    if element.ignore:
        return (PatternToken.ignore(), element.importance)
    token = tokenizer.str_to_id(element.token)
    return (PatternToken.regular(token), element.importance)


def _line_to_pattern(tokenizer: Tokenizer, line: Line) -> Pattern:
    activating = line[0]
    assert not activating.ignore
    assert not activating.is_end
    activating_token = tokenizer.str_to_id(activating.token)
    activating_importance = activating.importance
    activation = activating.activation
    context_tokens = [
        token for token in (_element_to_pattern_token(tokenizer, element) for element in line[1:]) if token is not None
    ]
    return Pattern.from_tokens(activating_token, activating_importance, context_tokens, activation)


def _model_from_lines(tokenizer: Tokenizer, lines: list[Line]) -> RustFeatureModel:
    patterns = [_line_to_pattern(tokenizer, line) for line in lines]
    return RustFeatureModel.from_patterns(patterns)


def _model_from_samples(
    tokenizer: Tokenizer, samples: list[list[Sample]], importance_threshold: float, activation_threshold: float
) -> RustFeatureModel:
    lines = neuron_model.samples_to_lines(samples, importance_threshold, activation_threshold)
    return _model_from_lines(tokenizer, lines)


def _str_token_to_pattern_token(tokenizer: Tokenizer, str_token: str) -> PatternToken:
    assert str_token != neuron_model.END_TOKEN
    assert str_token != neuron_model.ROOT_TOKEN
    if str_token == neuron_model.IGNORE_TOKEN:
        return PatternToken.ignore()
    return PatternToken.regular(tokenizer.str_to_id(str_token))


def _str_token_to_token(tokenizer: Tokenizer, str_token: str) -> Token:
    assert str_token != neuron_model.END_TOKEN
    assert str_token != neuron_model.ROOT_TOKEN
    assert str_token != neuron_model.IGNORE_TOKEN
    return Token.from_i32(tokenizer.str_to_id(str_token))


def _node_rust_to_py(tokenizer: Tokenizer, node: NeuronNode) -> RustFeatureModelNode:
    children = [
        (_str_token_to_pattern_token(tokenizer, str_token), _node_rust_to_py(tokenizer, child))
        for str_token, child in node.children.items()
        if str_token != neuron_model.END_TOKEN
    ]
    if node.value.is_end:
        return RustFeatureModelNode.from_children(children, node.value.importance, node.value.activation)
    else:
        return RustFeatureModelNode.from_children(children, node.value.importance)


@dataclass
class FeatureModel:
    _model: RustFeatureModel
    _tokenizer: Tokenizer

    @staticmethod
    def from_samples(
        tokenizer: Tokenizer, samples: list[list[Sample]], importance_threshold: float, activation_threshold: float
    ) -> "FeatureModel":
        return FeatureModel(
            _model_from_samples(tokenizer, samples, importance_threshold, activation_threshold), tokenizer
        )

    @staticmethod
    def from_model(tokenizer: Tokenizer, model: NeuronModel) -> "FeatureModel":
        trie_root = model.trie_root
        nodes = [
            (_str_token_to_token(tokenizer, str_token), _node_rust_to_py(tokenizer, child))
            for str_token, child in trie_root.children.items()
        ]
        return FeatureModel(RustFeatureModel.from_nodes(nodes), tokenizer)

    def __call__(self, tokens_arr: list[list[str]]) -> list[list[float]]:
        return self.forward(tokens_arr)

    def forward(self, tokens_arr: list[list[str]]) -> list[list[float]]:
        return self._model.forward([[self._tokenizer.str_to_id(token) for token in tokens] for tokens in tokens_arr])

    def forward_tokens(self, tokens_arr: list[list[int]]) -> list[list[float]]:
        return self._model.forward(tokens_arr)

    def graphviz(self) -> str:
        return self._model.graphviz(lambda token: self._tokenizer.id_to_str(token))

    def tokens(self) -> list[tuple[str, bool]]:
        return [(self._tokenizer.id_to_str(token), activating) for token, activating in self._model.tokens()]
