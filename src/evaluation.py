"""
Evaluation metrics for N-gram models.
"""

from __future__ import annotations

import math

def calculate_cross_entropy(
    model,
    dataset,
    tokenizer,
    epsilon: float = 1e-12,
) -> float:
    """
    Calculate character-level cross entropy.
    """

    total_loss = 0.0

    total_predictions = 0

    for text in dataset:

        tokens = (
            tokenizer.character_tokenize(
                text
            )
        )

        ngrams = model.generate_ngrams(
            tokens
        )

        for ngram in ngrams:

            context = "".join(
                ngram[:-1]
            )

            next_token = ngram[-1]

            probability = model.probability(
                context,
                next_token,
            )

            probability = max(
                probability,
                epsilon,
            )

            total_loss += (
                -math.log2(
                    probability
                )
            )

            total_predictions += 1

    if total_predictions == 0:

        return float("inf")

    return (
        total_loss
        / total_predictions
    )


def calculate_perplexity(
    cross_entropy: float,
) -> float:
    """
    Calculate perplexity.

        PP = 2^H
    """

    if math.isinf(
        cross_entropy
    ):

        return float("inf")

    return 2 ** cross_entropy


def evaluate_model(
    model,
    dataset,
    tokenizer,
) -> dict[str, float]:
    """
    Evaluate an N-gram model.
    """

    cross_entropy = (
        calculate_cross_entropy(
            model,
            dataset,
            tokenizer,
        )
    )

    perplexity = (
        calculate_perplexity(
            cross_entropy
        )
    )

    return {
        "cross_entropy": cross_entropy,
        "perplexity": perplexity,
    }