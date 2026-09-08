"""
Character-level tokenizer for Arabic text.
"""

from __future__ import annotations


class ArabicCharacterTokenizer:
    """
    Character-level tokenizer.

    Each Unicode character is treated as one token.

    Example:
        "سلام"
        →
        ["س", "ل", "ا", "م"]
    """

    UNKNOWN_TOKEN = "<UNK>"
    PAD_TOKEN = "<PAD>"

    def __init__(
        self,
        vocabulary: list[str] | None = None,
    ) -> None:
        """
        Initialize the tokenizer.

        Args:
            vocabulary:
                Optional pre-defined vocabulary.
                If None, vocabulary must be built later using fit().
        """
        self.vocabulary: list[str] = []

        self.token_to_index: dict[str, int] = {}
        self.index_to_token: dict[int, str] = {}

        self.vocabulary_size = 0

        self.pad_token_id: int | None = None
        self.unknown_token_id: int | None = None

        if vocabulary is not None:
            self.set_vocabulary(vocabulary)

    def character_tokenize(self, text: str) -> list[str]:
        """
        Split text into individual character tokens.

        Args:
            text: Input text.

        Returns:
            List of single-character strings.
        """
        if not isinstance(text, str):
            raise TypeError("text must be a string")

        return list(text)

    def join_text(self, tokens: list[str]) -> str:
        """
        Join character tokens into a string.

        Args:
            tokens: List of character tokens.

        Returns:
            Reconstructed text.
        """
        return "".join(tokens)

    def build_vocabulary(
        self,
        tokens: list[str],
    ) -> list[str]:
        """
        Build a sorted vocabulary from tokens.

        Duplicate tokens are removed.

        Args:
            tokens: List of character tokens.

        Returns:
            Sorted list of unique tokens.
        """
        unique_tokens = sorted(set(tokens))

        return unique_tokens

    def set_vocabulary(
        self,
        vocabulary: list[str],
    ) -> None:
        """
        Set vocabulary and create token/index mappings.

        Special tokens are automatically added if they are missing.

        Args:
            vocabulary: List of vocabulary tokens.
        """
        vocabulary = list(dict.fromkeys(vocabulary))

        if self.PAD_TOKEN not in vocabulary:
            vocabulary.insert(0, self.PAD_TOKEN)

        if self.UNKNOWN_TOKEN not in vocabulary:
            vocabulary.append(self.UNKNOWN_TOKEN)

        self.vocabulary = vocabulary

        self.vocabulary_size = len(self.vocabulary)

        self.token_to_index = {
            token: index
            for index, token in enumerate(self.vocabulary)
        }

        self.index_to_token = {
            index: token
            for index, token in enumerate(self.vocabulary)
        }

        self.pad_token_id = self.token_to_index[self.PAD_TOKEN]
        self.unknown_token_id = self.token_to_index[self.UNKNOWN_TOKEN]

    def fit(self, corpus: list[str]) -> None:
        """
        Build vocabulary from a text corpus.

        Args:
            corpus: List of text strings.
        """
        if isinstance(corpus, str):
            corpus = [corpus]

        all_tokens: list[str] = []

        for text in corpus:
            all_tokens.extend(
                self.character_tokenize(text)
            )

        vocabulary = self.build_vocabulary(all_tokens)

        self.set_vocabulary(vocabulary)

    def encode(self, text: str) -> list[int]:
        """
        Convert text into token IDs.

        Unknown characters are mapped to <UNK>.

        Args:
            text: Input text.

        Returns:
            List of integer token IDs.
        """
        if not self.token_to_index:
            raise RuntimeError(
                "Tokenizer has no vocabulary. "
                "Call fit() or set_vocabulary() first."
            )

        tokens = self.character_tokenize(text)

        indices = []

        for token in tokens:
            token_id = self.token_to_index.get(
                token,
                self.unknown_token_id,
            )

            indices.append(token_id)

        return indices

    def decode(
        self,
        indices: list[int],
        skip_special_tokens: bool = False,
    ) -> str:
        """
        Convert token IDs back into text.

        Args:
            indices: List of integer token IDs.
            skip_special_tokens:
                If True, <PAD> and <UNK> are omitted.

        Returns:
            Decoded text.
        """
        if not self.index_to_token:
            raise RuntimeError(
                "Tokenizer has no vocabulary. "
                "Call fit() or set_vocabulary() first."
            )

        tokens: list[str] = []

        for index in indices:
            token = self.index_to_token.get(
                index,
                self.UNKNOWN_TOKEN,
            )

            if skip_special_tokens and token in {
                self.PAD_TOKEN,
                self.UNKNOWN_TOKEN,
            }:
                continue

            tokens.append(token)

        return self.join_text(tokens)

    def token_id(self, token: str) -> int:
        """
        Return the integer ID of a token.

        Unknown tokens return the <UNK> ID.

        Args:
            token: Token.

        Returns:
            Token ID.
        """
        if self.unknown_token_id is None:
            raise RuntimeError(
                "Tokenizer has no vocabulary."
            )

        return self.token_to_index.get(
            token,
            self.unknown_token_id,
        )

    def token_from_id(self, index: int) -> str:
        """
        Return the token associated with an ID.

        Unknown IDs return <UNK>.

        Args:
            index: Token ID.

        Returns:
            Token string.
        """
        return self.index_to_token.get(
            index,
            self.UNKNOWN_TOKEN,
        )

    def __len__(self) -> int:
        """
        Return vocabulary size.
        """
        return self.vocabulary_size

    def __repr__(self) -> str:
        """
        Return a readable representation.
        """
        return (
            f"ArabicCharacterTokenizer("
            f"vocabulary_size={self.vocabulary_size})"
        )