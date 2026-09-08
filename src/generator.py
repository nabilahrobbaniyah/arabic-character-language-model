"""
Text generation utilities for the character-level N-gram model.
"""

from __future__ import annotations

import random

from src.ngram import CharacterNGramModel


class NGramTextGenerator:
    """
    Generate text using a trained CharacterNGramModel.
    """

    def __init__(
        self,
        model: CharacterNGramModel,
        tokenizer,
    ) -> None:
        """
        Initialize the text generator.

        Args:
            model:
                Trained N-gram model.

            tokenizer:
                Character tokenizer.
        """

        self.model = model
        self.tokenizer = tokenizer

    def _get_context(
        self,
        generated_tokens: list[str],
    ) -> str:
        """
        Extract the latest n-1 characters.

        Args:
            generated_tokens:
                Characters generated so far.

        Returns:
            Current context.
        """

        context_length = self.model.n - 1

        if len(generated_tokens) < context_length:
            return self.tokenizer.join_text(
                generated_tokens
            )

        return self.tokenizer.join_text(
            generated_tokens[-context_length:]
        )

    def _sample_random_token(
        self,
        probabilities: dict[str, float],
    ) -> str:
        """
        Sample a token using probability-weighted random sampling.

        Args:
            probabilities:
                Mapping from token to probability.

        Returns:
            Sampled token.
        """

        tokens = list(probabilities.keys())

        weights = list(probabilities.values())

        return random.choices(
            tokens,
            weights=weights,
            k=1,
        )[0]

    def _sample_greedy_token(
        self,
        probabilities: dict[str, float],
    ) -> str:
        """
        Select the token with the highest probability.

        Args:
            probabilities:
                Mapping from token to probability.

        Returns:
            Most likely token.
        """

        return max(
            probabilities,
            key=probabilities.get,
        )

    def _fallback_token(
        self,
    ) -> str | None:
        """
        Select a fallback token when a context is unseen.

        The most frequent character is used.

        Returns:
            Fallback character.
        """

        if not self.model.token_counts:
            return None

        return self.model.token_counts.most_common(
            1
        )[0][0]

    def generate(
        self,
        start_prompt: str,
        n_tokens: int,
        sampling_mode: str = "random",
    ) -> str:
        """
        Generate text from a starting prompt.

        Args:
            start_prompt:
                Initial text.

            n_tokens:
                Number of new characters to generate.

            sampling_mode:
                "random" or "greedy".

        Returns:
            Generated text including the original prompt.
        """

        if n_tokens < 0:
            raise ValueError(
                "n_tokens must be non-negative"
            )

        if sampling_mode not in {
            "random",
            "greedy",
        }:
            raise ValueError(
                "sampling_mode must be "
                "'random' or 'greedy'"
            )

        generated_tokens = (
            self.tokenizer.character_tokenize(
                start_prompt
            )
        )

        for _ in range(n_tokens):

            context = self._get_context(
                generated_tokens
            )

            probabilities = (
                self.model.get_next_token_probabilities(
                    context
                )
            )

            # If the context was never observed,
            # use the most frequent character.
            if not probabilities:

                fallback = self._fallback_token()

                if fallback is None:
                    break

                generated_tokens.append(
                    fallback
                )

                continue

            if sampling_mode == "random":

                next_token = (
                    self._sample_random_token(
                        probabilities
                    )
                )

            else:

                next_token = (
                    self._sample_greedy_token(
                        probabilities
                    )
                )

            generated_tokens.append(
                next_token
            )

        return self.tokenizer.join_text(
            generated_tokens
        )