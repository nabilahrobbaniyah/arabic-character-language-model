"""
Character-level tokenizer for Arabic text.
"""

from __future__ import annotations


class SimpleArabicCharacterTokenizer:
    """
    Simple character-level tokenizer.

    Each Unicode character is treated as one token.
    """

    def __init__(self):
        pass

    def character_tokenize(
        self,
        text: str,
    ) -> list[str]:
        """
        Split text into individual character tokens.

        Example:

            "سلام"

        becomes:

            ["س", "ل", "ا", "م"]
        """

        if not isinstance(text, str):
            raise TypeError(
                "text must be a string."
            )

        return list(text)

    def join_text(
        self,
        tokens: list[str],
    ) -> str:
        """
        Join character tokens into a string.
        """

        return "".join(tokens)


class EnhancedTokenizer(
    SimpleArabicCharacterTokenizer
):
    """
    Character tokenizer with vocabulary,
    encode, and decode functionality.
    """

    UNKNOWN_TOKEN = "<UNK>"
    PAD_TOKEN = "<PAD>"

    def __init__(
        self,
        corpus: list[str] | None = None,
        vocabulary: list[str] | None = None,
    ):
        super().__init__()

        if vocabulary is None:

            if corpus is None:
                corpus = []

            if isinstance(corpus, str):
                corpus = [corpus]

            tokens = []

            for text in corpus:
                tokens.extend(
                    self.character_tokenize(text)
                )

            vocabulary = self.build_vocabulary(
                tokens
            )

            self.vocabulary = (
                [self.PAD_TOKEN]
                + vocabulary
                + [self.UNKNOWN_TOKEN]
            )

        else:

            self.vocabulary = vocabulary

        self.vocabulary_size = len(
            self.vocabulary
        )

        self.token_to_index = {}

        self.index_to_token = {}

        for index, token in enumerate(
            self.vocabulary
        ):

            self.token_to_index[
                token
            ] = index

            self.index_to_token[
                index
            ] = token

        self.pad_token_id = (
            self.token_to_index[
                self.PAD_TOKEN
            ]
        )

        self.unknown_token_id = (
            self.token_to_index[
                self.UNKNOWN_TOKEN
            ]
        )

    def build_vocabulary(
        self,
        tokens: list[str],
    ) -> list[str]:
        """
        Build a sorted vocabulary.
        """

        return sorted(
            list(set(tokens))
        )

    def encode(
        self,
        text: str,
    ) -> list[int]:
        """
        Convert text into token IDs.
        """

        indices = []

        tokens = self.character_tokenize(
            text
        )

        for token in tokens:

            index = self.token_to_index.get(
                token,
                self.unknown_token_id,
            )

            indices.append(index)

        return indices

    def decode(
        self,
        indices: int | list[int],
    ) -> str:
        """
        Convert token IDs back into text.
        """

        if isinstance(indices, int):
            indices = [indices]

        tokens = []

        for index in indices:

            token = self.index_to_token.get(
                index,
                self.UNKNOWN_TOKEN,
            )

            tokens.append(token)

        return self.join_text(tokens)