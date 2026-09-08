"""
Evaluation utilities for character-level N-gram language models.

Metrics:
    - Cross entropy
    - Perplexity
    - Average negative log likelihood
"""

from __future__ import annotations

import math

from src.ngram import CharacterNGramModel


def calculate_cross_entropy(
    model: CharacterNGramModel,
    dataset: list[str],
    tokenizer,
    epsilon: float = 1e-12,
) -> float:
    """
    Calculate character-level cross entropy.

    Cross entropy:

        H = -1/N * sum(log2(P(token | context)))

    Args:
        model:
            Trained N-gram model.

        dataset:
            Evaluation dataset.

        tokenizer:
            Character tokenizer.

        epsilon:
            Small value used to avoid log(0).

    Returns:
        Average cross entropy in bits.
    """

    total_negative_log_probability = 0.0

    total_predictions = 0

    for text in dataset:

        tokens = tokenizer.character_tokenize(
            text
        )

        if len(tokens) < model.n:
            continue

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

            # Avoid log(0).
            probability = max(
                probability,
                epsilon,
            )

            negative_log_probability = (
                -math.log2(probability)
            )

            total_negative_log_probability += (
                negative_log_probability
            )

            total_predictions += 1

    if total_predictions == 0:
        return float("inf")

    return (
        total_negative_log_probability
        / total_predictions
    )


def calculate_perplexity(
    cross_entropy: float,
) -> float:
    """
    Calculate perplexity from cross entropy.

    Perplexity:

        PP = 2^H

    where H is cross entropy in bits.

    Args:
        cross_entropy:
            Cross entropy in bits.

    Returns:
        Perplexity.
    """

    if math.isinf(cross_entropy):
        return float("inf")

    return 2 ** cross_entropy


def evaluate_model(
    model: CharacterNGramModel,
    dataset: list[str],
    tokenizer,
) -> dict[str, float]:
    """
    Calculate all evaluation metrics for one model.

    Returns:
        Dictionary containing:

            cross_entropy
            perplexity
    """

    cross_entropy = calculate_cross_entropy(
        model,
        dataset,
        tokenizer,
    )

    perplexity = calculate_perplexity(
        cross_entropy
    )

    return {
        "cross_entropy": cross_entropy,
        "perplexity": perplexity,
    }


def calculate_average_log_probability(
    model: CharacterNGramModel,
    dataset: list[str],
    tokenizer,
    epsilon: float = 1e-12,
) -> float:
    """
    Calculate average log probability.

    This is useful for debugging and analysis.

    Returns:
        Average log2 probability.
    """

    total_log_probability = 0.0

    total_predictions = 0

    for text in dataset:

        tokens = tokenizer.character_tokenize(
            text
        )

        if len(tokens) < model.n:
            continue

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

            total_log_probability += (
                math.log2(probability)
            )

            total_predictions += 1

    if total_predictions == 0:
        return float("-inf")

    return (
        total_log_probability
        / total_predictions
    )


def calculate_unknown_context_rate(
    model: CharacterNGramModel,
    dataset: list[str],
    tokenizer,
) -> float:
    """
    Calculate the percentage of evaluation contexts
    that were never observed during training.

    Returns:
        Unknown context rate between 0 and 1.
    """

    unknown_contexts = 0
    total_contexts = 0

    for text in dataset:

        tokens = tokenizer.character_tokenize(
            text
        )

        if len(tokens) < model.n:
            continue

        ngrams = model.generate_ngrams(
            tokens
        )

        for ngram in ngrams:

            context = "".join(
                ngram[:-1]
            )

            total_contexts += 1

            if context not in model.probabilities:

                unknown_contexts += 1

    if total_contexts == 0:
        return 0.0

    return (
        unknown_contexts
        / total_contexts
    )