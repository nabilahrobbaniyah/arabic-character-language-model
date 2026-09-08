"""
Text generation utilities.
"""

from __future__ import annotations

import random


def generate_text(
    start_prompt: str,
    n_tokens: int,
    model,
    tokenizer,
    sampling_mode: str = "random",
    seed: int | None = None,
) -> str:
    """
    Generate text from a trained N-gram model.

    Args:
        start_prompt:
            Initial text.

        n_tokens:
            Number of characters to generate.

        model:
            Trained CharacterNGramModel.

        tokenizer:
            Character tokenizer.

        sampling_mode:
            "random" or "greedy".

        seed:
            Optional random seed.

    Returns:
        Generated text.
    """

    if n_tokens < 0:

        raise ValueError(
            "n_tokens must be >= 0."
        )

    if sampling_mode not in {
        "random",
        "greedy",
    }:

        raise ValueError(
            "sampling_mode must be "
            "'random' or 'greedy'."
        )

    if seed is not None:

        random.seed(seed)

    generated_tokens = (
        tokenizer.character_tokenize(
            start_prompt
        )
    )

    for _ in range(n_tokens):

        # Take the most recent n-1 characters.
        context_tokens = (
            generated_tokens[
                -(model.n - 1):
            ]
        )

        context = tokenizer.join_text(
            context_tokens
        )

        probabilities = (
            model.get_next_token_probabilities(
                context
            )
        )

        # Unknown context

        if not probabilities:

            vocabulary = (
                model.get_vocabulary()
            )

            if not vocabulary:
                break

            if sampling_mode == "greedy":

                next_token = max(
                    vocabulary,
                    key=lambda token:
                    model.get_marginal_probability(
                        token
                    ),
                )

            else:

                weights = [
                    model.get_marginal_probability(
                        token
                    )
                    for token in vocabulary
                ]

                if sum(weights) == 0:

                    next_token = random.choice(
                        vocabulary
                    )

                else:

                    next_token = (
                        random.choices(
                            vocabulary,
                            weights=weights,
                            k=1,
                        )[0]
                    )

            generated_tokens.append(
                next_token
            )

            continue

        # Greedy sampling

        if sampling_mode == "greedy":

            next_token = max(
                probabilities,
                key=probabilities.get,
            )

        # Random weighted sampling

        else:

            tokens = list(
                probabilities.keys()
            )

            weights = list(
                probabilities.values()
            )

            next_token = (
                random.choices(
                    tokens,
                    weights=weights,
                    k=1,
                )[0]
            )

        generated_tokens.append(
            next_token
        )

    return tokenizer.join_text(
        generated_tokens
    )