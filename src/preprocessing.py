"""
Dataset loading and preprocessing utilities.
"""

from __future__ import annotations

import re
from pathlib import Path


def load_text_file(
    path: str | Path,
) -> list[str]:
    """
    Load an Arabic text file.

    Each non-empty line becomes one dataset sample.
    """

    path = Path(path)

    if not path.exists():

        raise FileNotFoundError(
            f"Dataset not found: {path}"
        )

    text = path.read_text(
        encoding="utf-8"
    )

    lines = text.splitlines()

    dataset = [
        line.strip()
        for line in lines
        if line.strip()
    ]

    return dataset


def clean_text(
    text: str,
) -> str:
    """
    Basic Arabic text cleaning.

    The goal is to keep Arabic characters,
    spaces, and common punctuation.
    """

    if not isinstance(text, str):

        raise TypeError(
            "text must be a string."
        )

    # Normalize whitespace.
    text = re.sub(
        r"\s+",
        " ",
        text,
    )

    # Remove leading/trailing whitespace.
    text = text.strip()

    return text


def clean_dataset(
    dataset: list[str],
) -> list[str]:
    """
    Clean all samples in a dataset.
    """

    cleaned = []

    for text in dataset:

        text = clean_text(text)

        if text:

            cleaned.append(text)

    return cleaned