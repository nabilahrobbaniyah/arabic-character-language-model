"""
Text preprocessing utilities for the Arabic character-level language model.
"""

from __future__ import annotations

import re
import unicodedata
from pathlib import Path


# Arabic Unicode ranges commonly used in text.
ARABIC_CHARACTERS = re.compile(
    r"[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF]"
)

# Arabic diacritics / tashkeel.
ARABIC_DIACRITICS = re.compile(
    r"[\u0610-\u061A\u064B-\u065F\u0670\u06D6-\u06ED]"
)


def normalize_unicode(text: str) -> str:
    """
    Normalize Unicode representation of the input text.

    NFC is used so that visually identical characters are represented
    consistently when possible.

    Args:
        text: Input text.

    Returns:
        Unicode-normalized text.
    """
    return unicodedata.normalize("NFC", text)


def remove_diacritics(text: str) -> str:
    """
    Remove Arabic diacritics/tashkeel from text.

    Args:
        text: Input Arabic text.

    Returns:
        Text without Arabic diacritics.
    """
    return ARABIC_DIACRITICS.sub("", text)


def normalize_arabic_characters(text: str) -> str:
    """
    Normalize common Arabic character variants.

    The goal is to reduce unnecessary vocabulary differences.

    Normalizations:
        أ → ا
        إ → ا
        آ → ا
        ٱ → ا
        ى → ي
        ة is preserved.

    Args:
        text: Input Arabic text.

    Returns:
        Normalized Arabic text.
    """
    replacements = {
        "أ": "ا",
        "إ": "ا",
        "آ": "ا",
        "ٱ": "ا",
        "ى": "ي",
    }

    for old, new in replacements.items():
        text = text.replace(old, new)

    return text


def normalize_whitespace(text: str) -> str:
    """
    Normalize whitespace.

    Multiple spaces, tabs, and line breaks are collapsed into a
    single space.

    Args:
        text: Input text.

    Returns:
        Text with normalized whitespace.
    """
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def clean_text(
    text: str,
    *,
    remove_diacritics_flag: bool = True,
    normalize_characters: bool = True,
) -> str:
    """
    Clean and normalize Arabic text.

    Processing steps:
        1. Unicode normalization.
        2. Remove Arabic diacritics if requested.
        3. Normalize common Arabic character variants.
        4. Normalize whitespace.
        5. Remove control characters.

    Args:
        text: Input text.
        remove_diacritics_flag: Whether to remove Arabic diacritics.
        normalize_characters: Whether to normalize common Arabic variants.

    Returns:
        Cleaned text.
    """
    if not isinstance(text, str):
        raise TypeError("text must be a string")

    # Unicode normalization.
    text = normalize_unicode(text)

    # Remove control characters except whitespace.
    text = "".join(
        character
        for character in text
        if not unicodedata.category(character).startswith("C")
        or character in "\n\t\r"
    )

    # Remove Arabic diacritics.
    if remove_diacritics_flag:
        text = remove_diacritics(text)

    # Normalize Arabic character variants.
    if normalize_characters:
        text = normalize_arabic_characters(text)

    # Normalize spaces.
    text = normalize_whitespace(text)

    return text


def clean_dataset(dataset: list[str]) -> list[str]:
    """
    Clean every text entry in a dataset.

    Empty entries are removed.

    Args:
        dataset: List of raw text strings.

    Returns:
        List of cleaned non-empty strings.
    """
    cleaned_dataset = []

    for text in dataset:
        cleaned = clean_text(text)

        if cleaned:
            cleaned_dataset.append(cleaned)

    return cleaned_dataset


def load_text_file(path: str | Path) -> list[str]:
    """
    Load a text file line by line.

    Each non-empty line becomes one dataset entry.

    Args:
        path: Path to the text file.

    Returns:
        List of text entries.
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")

    with path.open("r", encoding="utf-8") as file:
        lines = file.readlines()

    return [line.strip() for line in lines if line.strip()]


def save_text_file(
    dataset: list[str],
    path: str | Path,
) -> None:
    """
    Save dataset entries to a UTF-8 text file.

    Each entry is written on a separate line.

    Args:
        dataset: List of text entries.
        path: Output path.
    """
    path = Path(path)

    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:
        for text in dataset:
            file.write(text + "\n")


def prepare_dataset(
    input_path: str | Path,
    output_path: str | Path,
) -> list[str]:
    """
    Load, clean, and save a dataset.

    Args:
        input_path: Path to raw dataset.
        output_path: Path for cleaned dataset.

    Returns:
        Cleaned dataset.
    """
    raw_dataset = load_text_file(input_path)

    cleaned_dataset = clean_dataset(raw_dataset)

    save_text_file(cleaned_dataset, output_path)

    return cleaned_dataset