"""
Character-level N-gram language model.
"""

from __future__ import annotations

from collections import Counter, defaultdict


class CharacterNGramModel:
    """
    Character-level N-gram language model.

    n = 2 -> Bigram
    n = 3 -> Trigram
    n = 4 -> 4-gram
    """

    def __init__(
        self,
        n: int = 3,
    ):
        if not isinstance(n, int):

            raise TypeError(
                "n must be an integer."
            )

        if n < 2:

            raise ValueError(
                "n must be >= 2."
            )

        self.n = n

        self.counts = {}

        self.probabilities = {}

        self.token_counts = Counter()

        self.total_tokens = 0

    def generate_ngrams(
        self,
        tokens: list[str],
    ) -> list[tuple[str, ...]]:
        """
        Generate character n-grams.
        """

        if len(tokens) < self.n:
            return []

        ngrams = []

        for i in range(
            len(tokens) - self.n + 1
        ):

            ngrams.append(
                tuple(
                    tokens[
                        i:i + self.n
                    ]
                )
            )

        return ngrams

    def get_contexts(self) -> list[str]:
        return sorted(self.probabilities.keys())

    def count_tokens(
        self,
        dataset: list[str],
        tokenizer,
    ) -> Counter[str]:
        """
        Count individual character tokens.
        """

        counts = Counter()

        for text in dataset:

            tokens = (
                tokenizer.character_tokenize(
                    text
                )
            )

            counts.update(tokens)

        return counts

    def count_ngrams(
        self,
        dataset: list[str],
        tokenizer,
    ):
        """
        Count n-grams.

        Returns:

            context -> next character counts
        """

        ngram_counts = defaultdict(
            Counter
        )

        for text in dataset:

            tokens = (
                tokenizer.character_tokenize(
                    text
                )
            )

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

        return dict(
            ngram_counts
        )

    def fit(
        self,
        dataset: list[str],
        tokenizer,
    ) -> None:
        """
        Train the N-gram model.
        """

        self.token_counts = (
            self.count_tokens(
                dataset,
                tokenizer,
            )
        )

        self.total_tokens = sum(
            self.token_counts.values()
        )

        self.counts = (
            self.count_ngrams(
                dataset,
                tokenizer,
            )
        )

        self.probabilities = {}

        for context, next_tokens in (
            self.counts.items()
        ):

            total = sum(
                next_tokens.values()
            )

            self.probabilities[
                context
            ] = {}

            for token, count in (
                next_tokens.items()
            ):

                self.probabilities[
                    context
                ][token] = (
                    count / total
                )

    def get_next_token_probabilities(
        self,
        context: str,
    ) -> dict[str, float]:
        """
        Return conditional probabilities
        for a context.
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
        Return P(token).
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
        Return:

            P(token | context)

        If the context is unseen,
        fall back to marginal probability.
        """

        probabilities = (
            self.get_next_token_probabilities(
                context
            )
        )

        if token in probabilities:

            return probabilities[token]

        return self.get_marginal_probability(
            token
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

    def get_vocabulary(self):
        """
        Return model vocabulary.
        """

        return sorted(
            self.token_counts.keys()
        )

    def __len__(self):
        """
        Return number of observed contexts.
        """

        return len(
            self.probabilities
        )

    def __repr__(self):

        return (
            "CharacterNGramModel("
            f"n={self.n}, "
            f"contexts={len(self)})"
        )