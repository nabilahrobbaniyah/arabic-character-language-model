"""
Character-level N-gram language model.

Supports:
    - Bigram
    - Trigram
    - Higher-order N-grams
    - N-gram counting
    - Conditional probability
    - Marginal probability fallback
"""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import DefaultDict


class CharacterNGramModel:
    """
    Character-level N-gram language model.

    For n=2:

        P(character_t | character_t-1)

    For n=3:

        P(character_t | character_t-2, character_t-1)

    For n=4:

        P(character_t | character_t-3,
                         character_t-2,
                         character_t-1)
    """

    def __init__(self, n: int = 3) -> None:

        if not isinstance(n, int):
            raise TypeError("n must be an integer")

        if n < 2:
            raise ValueError("n must be at least 2")

        self.n = n

        # context -> next-token counts
        self.counts: dict[str, Counter[str]] = {}

        # context -> next-token probabilities
        self.probabilities: dict[str, dict[str, float]] = {}

        # Individual character frequencies
        self.token_counts: Counter[str] = Counter()

        # Total number of characters
        self.total_tokens: int = 0

    # N-GRAM GENERATION

    def generate_ngrams(
        self,
        tokens: list[str],
    ) -> list[tuple[str, ...]]:
        """
        Generate N-grams from character tokens.

        Example:

            tokens = ["a", "b", "c", "d"]
            n = 3

        Result:

            [
                ("a", "b", "c"),
                ("b", "c", "d")
            ]
        """

        if len(tokens) < self.n:
            return []

        ngrams = []

        for i in range(len(tokens) - self.n + 1):

            ngram = tuple(
                tokens[i:i + self.n]
            )

            ngrams.append(ngram)

        return ngrams

    def generate_ngrams_from_text(
        self,
        text: str,
        tokenizer,
    ) -> list[tuple[str, ...]]:
        """
        Tokenize text and generate N-grams.
        """

        tokens = tokenizer.character_tokenize(text)

        return self.generate_ngrams(tokens)

    # COUNTING

    def count_tokens(
        self,
        dataset: list[str],
        tokenizer,
    ) -> Counter[str]:
        """
        Count individual character tokens.
        """

        token_counts = Counter()

        for text in dataset:

            tokens = tokenizer.character_tokenize(text)

            token_counts.update(tokens)

        return token_counts

    def count_ngrams(
        self,
        dataset: list[str],
        tokenizer,
    ) -> dict[str, Counter[str]]:
        """
        Count N-grams.

        Returns:

            context -> Counter(next_token)
        """

        ngram_counts: DefaultDict[
            str,
            Counter[str]
        ] = defaultdict(Counter)

        for text in dataset:

            tokens = tokenizer.character_tokenize(text)

            ngrams = self.generate_ngrams(
                tokens
            )

            for ngram in ngrams:

                context = "".join(
                    ngram[:-1]
                )

                next_token = ngram[-1]

                ngram_counts[
                    context
                ][next_token] += 1

        return dict(ngram_counts)

    # TRAINING

    def fit(
        self,
        dataset: list[str],
        tokenizer,
    ) -> None:
        """
        Train the N-gram model.
        """

        # Count individual characters.
        self.token_counts = self.count_tokens(
            dataset,
            tokenizer,
        )

        self.total_tokens = sum(
            self.token_counts.values()
        )

        # Count N-grams.
        self.counts = self.count_ngrams(
            dataset,
            tokenizer,
        )

        # Convert counts to conditional probabilities.
        self.probabilities = {}

        for context, next_tokens in self.counts.items():

            context_total = sum(
                next_tokens.values()
            )

            self.probabilities[
                context
            ] = {}

            for token, count in next_tokens.items():

                probability = (
                    count / context_total
                )

                self.probabilities[
                    context
                ][token] = probability

    # PROBABILITY

    def get_next_token_probabilities(
        self,
        context: str,
    ) -> dict[str, float]:
        """
        Return P(next_token | context).
        """

        return self.probabilities.get(
            context,
            {},
        )

    def get_marginal_probability(
        self,
        token: str,
    ) -> float:
        """
        Calculate P(token).

        This is the marginal probability of a character.
        """

        if self.total_tokens == 0:
            return 0.0

        return (
            self.token_counts[token]
            / self.total_tokens
        )

    def probability(
        self,
        context: str,
        token: str,
    ) -> float:
        """
        Calculate:

            P(token | context)

        If the context has never been seen,
        fall back to P(token).
        """

        probabilities = (
            self.get_next_token_probabilities(
                context
            )
        )

        if token in probabilities:

            return probabilities[token]

        # Backoff to marginal probability.
        return self.get_marginal_probability(
            token
        )

    # INFORMATION

    def get_context_count(
        self,
        context: str,
    ) -> int:
        """
        Return the total number of observations
        for a context.
        """

        if context not in self.counts:
            return 0

        return sum(
            self.counts[context].values()
        )

    def most_likely_next_token(
        self,
        context: str,
    ) -> str | None:
        """
        Return the most likely next character.
        """

        probabilities = (
            self.get_next_token_probabilities(
                context
            )
        )

        if not probabilities:
            return None

        return max(
            probabilities,
            key=probabilities.get,
        )

    def get_contexts(self) -> list[str]:
        """
        Return all observed contexts.
        """

        return list(
            self.probabilities.keys()
        )

    def get_vocabulary(self) -> list[str]:
        """
        Return all characters observed
        during training.
        """

        return sorted(
            self.token_counts.keys()
        )

    def __len__(self) -> int:
        """
        Number of observed contexts.
        """

        return len(
            self.probabilities
        )

    def __repr__(self) -> str:

        return (
            f"CharacterNGramModel("
            f"n={self.n}, "
            f"contexts={len(self)})"
        )