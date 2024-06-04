from typing import Callable

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
