"""
Tests for the character-level N-gram language model.
"""

from src.tokenizer import (
    ArabicCharacterTokenizer,
)

from src.ngram import (
    CharacterNGramModel,
)

from src.generator import (
    NGramTextGenerator,
)


def create_test_tokenizer():
    """
    Create a small tokenizer for testing.
    """

    dataset = [
        "اباباب",
        "اباب",
    ]

    tokenizer = ArabicCharacterTokenizer()

    tokenizer.fit(dataset)

    return tokenizer


def test_bigram_generation():
    """
    Test Bigram generation.
    """

    tokenizer = create_test_tokenizer()

    model = CharacterNGramModel(
        n=2
    )

    tokens = tokenizer.character_tokenize(
        "اب"
    )

    ngrams = model.generate_ngrams(
        tokens
    )

    assert ngrams == [
        ("ا", "ب"),
    ]


def test_trigram_generation():
    """
    Test Trigram generation.
    """

    tokenizer = create_test_tokenizer()

    model = CharacterNGramModel(
        n=3
    )

    tokens = tokenizer.character_tokenize(
        "ابا"
    )

    ngrams = model.generate_ngrams(
        tokens
    )

    assert ngrams == [
        ("ا", "ب", "ا"),
    ]


def test_fourgram_generation():
    """
    Test 4-gram generation.
    """

    tokenizer = create_test_tokenizer()

    model = CharacterNGramModel(
        n=4
    )

    tokens = tokenizer.character_tokenize(
        "اباب"
    )

    ngrams = model.generate_ngrams(
        tokens
    )

    assert ngrams == [
        ("ا", "ب", "ا", "ب"),
    ]


def test_ngram_short_sequence():
    """
    A sequence shorter than n should produce no N-grams.
    """

    tokenizer = create_test_tokenizer()

    model = CharacterNGramModel(
        n=4
    )

    tokens = tokenizer.character_tokenize(
        "اب"
    )

    ngrams = model.generate_ngrams(
        tokens
    )

    assert ngrams == []


def test_bigram_counts():
    """
    Test Bigram frequency counting.
    """

    dataset = [
        "abab",
    ]

    tokenizer = ArabicCharacterTokenizer()

    tokenizer.fit(dataset)

    model = CharacterNGramModel(
        n=2
    )

    model.fit(
        dataset,
        tokenizer,
    )

    assert model.counts["a"]["b"] == 2
    assert model.counts["b"]["a"] == 1


def test_probability():
    """
    Test conditional probability.
    """

    dataset = [
        "abab",
    ]

    tokenizer = ArabicCharacterTokenizer()

    tokenizer.fit(dataset)

    model = CharacterNGramModel(
        n=2
    )

    model.fit(
        dataset,
        tokenizer,
    )

    assert model.probability(
        "a",
        "b",
    ) == 1.0

    assert model.probability(
        "b",
        "a",
    ) == 1.0


def test_probability_sums_to_one():
    """
    Probabilities for an observed context should sum to 1.
    """

    dataset = [
        "ababab",
    ]

    tokenizer = ArabicCharacterTokenizer()

    tokenizer.fit(dataset)

    model = CharacterNGramModel(
        n=2
    )

    model.fit(
        dataset,
        tokenizer,
    )

    for context, probabilities in (
        model.probabilities.items()
    ):

        total_probability = sum(
            probabilities.values()
        )

        assert abs(
            total_probability - 1.0
        ) < 1e-9


def test_most_likely_next_token():
    """
    Test greedy prediction.
    """

    dataset = [
        "abababab",
        "abab",
        "abac",
    ]

    tokenizer = ArabicCharacterTokenizer()

    tokenizer.fit(dataset)

    model = CharacterNGramModel(
        n=2
    )

    model.fit(
        dataset,
        tokenizer,
    )

    assert (
        model.most_likely_next_token("a")
        == "b"
    )


def test_model_vocabulary():
    """
    Test model vocabulary.
    """

    dataset = [
        "ابج",
    ]

    tokenizer = ArabicCharacterTokenizer()

    tokenizer.fit(dataset)

    model = CharacterNGramModel(
        n=2
    )

    model.fit(
        dataset,
        tokenizer,
    )

    vocabulary = model.get_vocabulary()

    assert "ا" in vocabulary
    assert "ب" in vocabulary
    assert "ج" in vocabulary


def test_generator_keeps_prompt():
    """
    Generated text must contain the original prompt.
    """

    dataset = [
        "اباباباباب",
    ]

    tokenizer = ArabicCharacterTokenizer()

    tokenizer.fit(dataset)

    model = CharacterNGramModel(
        n=2
    )

    model.fit(
        dataset,
        tokenizer,
    )

    generator = NGramTextGenerator(
        model,
        tokenizer,
    )

    prompt = "ا"

    result = generator.generate(
        prompt,
        n_tokens=5,
        sampling_mode="greedy",
    )

    assert result.startswith(prompt)


def test_generator_length():
    """
    Generated text should contain prompt + n_tokens.
    """

    dataset = [
        "اباباباباب",
    ]

    tokenizer = ArabicCharacterTokenizer()

    tokenizer.fit(dataset)

    model = CharacterNGramModel(
        n=2
    )

    model.fit(
        dataset,
        tokenizer,
    )

    generator = NGramTextGenerator(
        model,
        tokenizer,
    )

    prompt = "ا"

    result = generator.generate(
        prompt,
        n_tokens=5,
        sampling_mode="greedy",
    )

    assert len(result) == len(prompt) + 5


def test_invalid_n():
    """
    N-gram size must be at least 2.
    """

    try:
        CharacterNGramModel(n=1)

        assert False

    except ValueError:
        assert True


def test_invalid_sampling_mode():
    """
    Generator should reject unsupported sampling modes.
    """

    dataset = [
        "abababab",
    ]

    tokenizer = ArabicCharacterTokenizer()

    tokenizer.fit(dataset)

    model = CharacterNGramModel(
        n=2
    )

    model.fit(
        dataset,
        tokenizer,
    )

    generator = NGramTextGenerator(
        model,
        tokenizer,
    )

    try:

        generator.generate(
            "a",
            5,
            sampling_mode="invalid",
        )

        assert False

    except ValueError:
        assert True