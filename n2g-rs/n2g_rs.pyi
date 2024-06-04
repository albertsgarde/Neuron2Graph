from typing import Callable

class Token:
    @staticmethod
    def from_i32(token: int) -> "Token":
        """
        Constructs a token from an integer.

        Args:
            token: The token to construct.
        """

class PatternToken:
    @staticmethod
    def ignore() -> "PatternToken":
        """
        Returns a pattern token that matches everything.
        """

    @staticmethod
    def regular(token: int) -> "PatternToken":
        """
        Returns a pattern token that matches single regular token.

        Args:
            token: The token to match.
        """

class Pattern:
    @staticmethod
    def from_tokens(
        activating_token: int,
        activating_importance: float,
        context: list[tuple[PatternToken, float]],
        activation: float,
    ) -> "Pattern":
        """
        Constructs a pattern from a list of tokens.

        Args:
            activating_token: The token on which the activation is measured. The last token in the pattern.
            activating_importance: The importance of the activating token.
            context: The context in reverse order. Each tuple is a token and its importance.
            activation: The predicted activation if this pattern matches.
        """

class FeatureModelNode:
    @staticmethod
    def from_children(
        children: list[tuple[PatternToken, "FeatureModelNode"]],
        importance: float,
        activation: float | None = None,
    ) -> "FeatureModelNode":
        """
        Constructs a feature model node from a list of children.

        Args:
            children: The children of the node.
            importance: The importance of the node.
            activation: The activation of the node.
        """

class FeatureModel:
    @staticmethod
    def from_patterns(
        patterns: list[Pattern],
    ) -> "FeatureModel":
        """
        Constructs a feature model from a list of patterns.

        Args:
            patterns: The patterns to use.
        """

    @staticmethod
    def from_nodes(
        nodes: list[tuple[Token, FeatureModelNode]],
    ) -> "FeatureModel":
        """
        Constructs a feature model from a list of nodes.

        Args:
            nodes: The nodes to use.
        """

    def __call__(self, tokens: list[list[int]]) -> list[list[float]]:
        """
        Predicts the activation for each token in each sample given the previous tokens as context.

        Args:
            tokens: The samples to predict activations for.
        """

    def forward(self, tokens: list[list[int]]) -> list[list[float]]:
        """
        Predicts the activation for each token in each sample given the previous tokens as context.

        Args:
            tokens: The samples to predict activations for.
        """

    def graphviz(self, token_to_str: Callable[[int], str]) -> str:
        """
        Returns a graphviz representation of the feature model.

        Args:
            token_to_str: A function that maps token ids to strings.
        """

    def predict_activation(self, tokens: list[int]) -> float:
        """
        Predicts the activation of the first token in the list given the rest as context in reverse order

        Args:
            tokens: sample to predict activation for in reverse order.
        """
    def tokens(self) -> list[tuple[int, bool]]:
        """
        Returns the tokens in the feature model and whether they are activating.
        """

    def to_json(self) -> str:
        """
        Returns a JSON representation of the feature model.
        """

    @staticmethod
    def from_json(json: str) -> "FeatureModel":
        """
        Constructs a feature model from a JSON representation.

        Args:
            json: The JSON representation to construct from.
        """

    def to_bin(self) -> bytes:
        """
        Returns a binary representation of the feature model.
        """

    @staticmethod
    def from_bin(bin: bytes) -> "FeatureModel":
        """
        Constructs a feature model from a binary representation.

        Args:
            bin: The binary representation to construct from.
        """
