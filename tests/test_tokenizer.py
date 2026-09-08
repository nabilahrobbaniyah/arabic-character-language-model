"""
Tests for ArabicCharacterTokenizer.
"""

from src.tokenizer import ArabicCharacterTokenizer


def test_character_tokenize():
    tokenizer = ArabicCharacterTokenizer()

    text = "الشمس طلعت"

    tokens = tokenizer.character_tokenize(text)

    assert tokens == [
        "ا",
        "ل",
        "ش",
        "م",
        "س",
        " ",
        "ط",
        "ل",
        "ع",
        "ت",
    ]


def test_join_text():
    tokenizer = ArabicCharacterTokenizer()

    tokens = [
        "ا",
        "ل",
        "ش",
        "م",
        "س",
    ]

    text = tokenizer.join_text(tokens)

    assert text == "الشمس"


def test_tokenize_and_join_are_inverse():
    tokenizer = ArabicCharacterTokenizer()

    text = "مرحبا بالعالم"

    tokens = tokenizer.character_tokenize(text)

    reconstructed_text = tokenizer.join_text(tokens)

    assert reconstructed_text == text


def test_vocabulary():
    tokenizer = ArabicCharacterTokenizer()

    corpus = [
        "ابا",
        "باب",
    ]

    tokenizer.fit(corpus)

    assert tokenizer.PAD_TOKEN in tokenizer.vocabulary
    assert tokenizer.UNKNOWN_TOKEN in tokenizer.vocabulary

    assert "ا" in tokenizer.vocabulary
    assert "ب" in tokenizer.vocabulary


def test_encode():
    tokenizer = ArabicCharacterTokenizer()

    tokenizer.fit(["اب"])

    encoded = tokenizer.encode("اب")

    assert len(encoded) == 2

    assert encoded[0] == tokenizer.token_to_index["ا"]
    assert encoded[1] == tokenizer.token_to_index["ب"]


def test_unknown_token():
    tokenizer = ArabicCharacterTokenizer()

    tokenizer.fit(["اب"])

    encoded = tokenizer.encode("ا ج")

    assert encoded[0] == tokenizer.token_to_index["ا"]
    assert encoded[1] == tokenizer.unknown_token_id
    assert encoded[2] == tokenizer.unknown_token_id


def test_decode():
    tokenizer = ArabicCharacterTokenizer()

    tokenizer.fit(["اب"])

    encoded = tokenizer.encode("اب")

    decoded = tokenizer.decode(encoded)

    assert decoded == "اب"


def test_encode_decode_round_trip():
    tokenizer = ArabicCharacterTokenizer()

    corpus = [
        "الشمس طلعت",
        "يوم جميل",
    ]

    tokenizer.fit(corpus)

    text = "الشمس طلعت"

    encoded = tokenizer.encode(text)
    decoded = tokenizer.decode(encoded)

    assert decoded == text


def test_unknown_id_decoding():
    tokenizer = ArabicCharacterTokenizer()

    tokenizer.fit(["اب"])

    decoded = tokenizer.decode([99999])

    assert decoded == tokenizer.UNKNOWN_TOKEN


def test_token_id():
    tokenizer = ArabicCharacterTokenizer()

    tokenizer.fit(["اب"])

    assert tokenizer.token_id("ا") == tokenizer.token_to_index["ا"]

    assert tokenizer.token_id("X") == tokenizer.unknown_token_id


def test_token_from_id():
    tokenizer = ArabicCharacterTokenizer()

    tokenizer.fit(["اب"])

    token_id = tokenizer.token_id("ا")

    assert tokenizer.token_from_id(token_id) == "ا"


def test_vocabulary_size():
    tokenizer = ArabicCharacterTokenizer()

    tokenizer.fit(["اب"])

    assert len(tokenizer) == tokenizer.vocabulary_size