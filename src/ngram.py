"""
Character-level N-gram language model.

This module provides:
    - N-gram generation
    - N-gram counting
    - Conditional probability calculation
    - Bigram, trigram, and higher-order N-gram models
"""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import DefaultDict


class CharacterNGramModel:
    """
    Character-level N-gram language model.

    Example for n=3:

        Context:
            "الش"

        Possible next character:
            "م"

        This represents:

            P(م | الش)
    """

    def __init__(self, n: int = 3) -> None:
        """
        Initialize the N-gram model.

        Args:
            n:
                Number of characters in each N-gram.

                n=2 -> bigram
                n=3 -> trigram
                n=4 -> 4-gram
        """

        if not isinstance(n, int):
            raise TypeError("n must be an integer")

        if n < 2:
            raise ValueError("n must be at least 2")

        self.n = n

        # Context -> Counter of possible next characters.
        #
        # Example:
        #
        # "ال" -> {"ش": 10, "ب": 4}
        #
        self.counts: dict[str, Counter[str]] = {}

        # Context -> probability distribution.
        #
        # Example:
        #
        # "ال" -> {"ش": 0.71, "ب": 0.29}
        #
        self.probabilities: dict[str, dict[str, float]] = {}

        # Overall character counts.
        self.token_counts: Counter[str] = Counter()

        # Total number of tokens.
        self.total_tokens: int = 0

    def generate_ngrams(
        self,
        tokens: list[str],
    ) -> list[tuple[str, ...]]:
        """
        Generate N-grams from a list of character tokens.

        Example:

            tokens = ["ا", "ب", "ج", "د"]
            n = 3

            result:

            [
                ("ا", "ب", "ج"),
                ("ب", "ج", "د")
            ]

        Args:
            tokens:
                List of character tokens.

        Returns:
            List of N-grams.
        """

        if len(tokens) < self.n:
            return []

        ngrams = []

        for i in range(len(tokens) - self.n + 1):
            ngram = tuple(tokens[i : i + self.n])
            ngrams.append(ngram)

        return ngrams

    def generate_ngrams_from_text(
        self,
        text: str,
        tokenizer,
    ) -> list[tuple[str, ...]]:
        """
        Tokenize text and generate N-grams.

        Args:
            text:
                Input text.

            tokenizer:
                Character tokenizer.

        Returns:
            List of N-grams.
        """

        tokens = tokenizer.character_tokenize(text)

        return self.generate_ngrams(tokens)

    def count_ngrams(
        self,
        dataset: list[str],
        tokenizer,
    ) -> dict[str, Counter[str]]:
        """
        Count N-grams in a dataset.

        The model stores:

            context -> next character -> count

        For example, with n=3:

            "الش" -> {
                "م": 10,
                "ي": 2
            }

        This means:

            "الشم" appeared 10 times
            "الشي" appeared 2 times

        Args:
            dataset:
                List of text strings.

            tokenizer:
                Character tokenizer.

        Returns:
            Dictionary containing N-gram counts.
        """

        ngram_counts: DefaultDict[str, Counter[str]] = defaultdict(Counter)

        for text in dataset:

            tokens = tokenizer.character_tokenize(text)

            ngrams = self.generate_ngrams(tokens)

            for ngram in ngrams:

                context = "".join(ngram[:-1])

                next_token = ngram[-1]

                ngram_counts[context][next_token] += 1

        return dict(ngram_counts)

    def count_tokens(
        self,
        dataset: list[str],
        tokenizer,
    ) -> Counter[str]:
        """
        Count individual character tokens in the dataset.

        Args:
            dataset:
                List of text strings.

            tokenizer:
                Character tokenizer.

        Returns:
            Counter containing character frequencies.
        """

        token_counts = Counter()

        for text in dataset:

            tokens = tokenizer.character_tokenize(text)

            token_counts.update(tokens)

        return token_counts

    def fit(
        self,
        dataset: list[str],
        tokenizer,
    ) -> None:
        """
        Train the N-gram model on a dataset.

        Steps:

            1. Count characters.
            2. Count N-grams.
            3. Convert counts into probabilities.

        Args:
            dataset:
                Training dataset.

            tokenizer:
                Character tokenizer.
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

        # Convert counts into probabilities.
        self.probabilities = {}

        for context, next_tokens in self.counts.items():

            context_total = sum(
                next_tokens.values()
            )

            self.probabilities[context] = {}

            for token, count in next_tokens.items():

                probability = count / context_total

                self.probabilities[context][token] = probability

    def get_next_token_probabilities(
        self,
        context: str,
    ) -> dict[str, float]:
        """
        Get probabilities of possible next characters.

        Args:
            context:
                Character context of length n-1.

        Returns:
            Dictionary mapping next characters to probabilities.

        Example:

            {
                "ا": 0.2,
                "م": 0.5,
                "ن": 0.3
            }
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
        Calculate the marginal probability of a character.

        P(token)

        Args:
            token:
                Character token.

        Returns:
            Probability of the character.
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
        Get P(token | context).

        If the context has not been observed,
        return the marginal probability P(token).

        Args:
            context:
                Previous n-1 characters.

            token:
                Candidate next character.

        Returns:
            Conditional probability.
        """

        context_probabilities = (
            self.get_next_token_probabilities(
                context
            )
        )

        if token in context_probabilities:

            return context_probabilities[token]

        # Backoff to marginal probability.
        return self.get_marginal_probability(
            token
        )

    def get_context_count(
        self,
        context: str,
    ) -> int:
        """
        Return the total number of observations for a context.

        Args:
            context:
                N-1 character context.

        Returns:
            Number of occurrences.
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

        Args:
            context:
                N-1 character context.

        Returns:
            Most likely character, or None if no probability exists.
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

        Returns:
            List of context strings.
        """

        return list(self.probabilities.keys())

    def get_vocabulary(self) -> list[str]:
        """
        Return all characters observed in the training data.

        Returns:
            List of character tokens.
        """

        return sorted(
            self.token_counts.keys()
        )

    def __len__(self) -> int:
        """
        Return the number of observed contexts.
        """

        return len(self.probabilities)

    def __repr__(self) -> str:
        """
        Return a readable representation.
        """

        return (
            f"CharacterNGramModel("
            f"n={self.n}, "
            f"contexts={len(self)})"
        )