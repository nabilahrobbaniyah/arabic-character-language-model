"""
Tests for N-gram evaluation metrics.
"""

from src.tokenizer import (
    ArabicCharacterTokenizer,
)

from src.ngram import (
    CharacterNGramModel,
)

from src.evaluation import (
    calculate_cross_entropy,
    calculate_perplexity,
    evaluate_model,
    calculate_unknown_context_rate,
)


def create_model():

    dataset = [
        "abababab",
        "abababab",
        "abababab",
    ]

    tokenizer = (
        ArabicCharacterTokenizer()
    )

    tokenizer.fit(
        dataset
    )

    model = CharacterNGramModel(
        n=2
    )

    model.fit(
        dataset,
        tokenizer,
    )

    return model, tokenizer


def test_cross_entropy_is_positive():

    model, tokenizer = (
        create_model()
    )

    dataset = [
        "ababab",
    ]

    cross_entropy = (
        calculate_cross_entropy(
            model,
            dataset,
            tokenizer,
        )
    )

    assert cross_entropy >= 0


def test_perplexity_from_cross_entropy():

    cross_entropy = 3.0

    perplexity = (
        calculate_perplexity(
            cross_entropy
        )
    )

    assert perplexity == 8.0


def test_perplexity_is_at_least_one():

    model, tokenizer = (
        create_model()
    )

    metrics = evaluate_model(
        model,
        ["ababab"],
        tokenizer,
    )

    assert metrics[
        "perplexity"
    ] >= 1.0


def test_evaluation_returns_metrics():

    model, tokenizer = (
        create_model()
    )

    metrics = evaluate_model(
        model,
        ["ababab"],
        tokenizer,
    )

    assert "cross_entropy" in metrics

    assert "perplexity" in metrics


def test_unknown_context_rate_range():

    model, tokenizer = (
        create_model()
    )

    rate = (
        calculate_unknown_context_rate(
            model,
            ["ababab"],
            tokenizer,
        )
    )

    assert 0.0 <= rate <= 1.0


def test_cross_entropy_not_nan():

    model, tokenizer = (
        create_model()
    )

    cross_entropy = (
        calculate_cross_entropy(
            model,
            ["ababab"],
            tokenizer,
        )
    )

    assert cross_entropy == cross_entropy


def test_model_probability():

    model, tokenizer = (
        create_model()
    )

    probability = model.probability(
        "a",
        "b",
    )

    assert probability > 0.0
    assert probability <= 1.0