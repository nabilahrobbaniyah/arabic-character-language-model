"""
Train and test Arabic character-level N-gram models.
"""

from __future__ import annotations

from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.preprocessing import (
    clean_dataset,
    load_text_file,
)

from src.tokenizer import (
    ArabicCharacterTokenizer,
)

from src.ngram import (
    CharacterNGramModel,
)

from src.generator import (
    NGramTextGenerator,
)


def main() -> None:
    """
    Train Bigram, Trigram, and 4-gram models.
    """

    dataset_path = (
        PROJECT_ROOT
        / "data"
        / "raw"
        / "arabic_sample.txt"
    )

    # --------------------------------------------------
    # 1. Load dataset
    # --------------------------------------------------

    dataset = load_text_file(
        dataset_path
    )

    # --------------------------------------------------
    # 2. Clean dataset
    # --------------------------------------------------

    dataset = clean_dataset(
        dataset
    )

    print(
        f"Dataset size: {len(dataset)}"
    )

    # --------------------------------------------------
    # 3. Build tokenizer
    # --------------------------------------------------

    tokenizer = ArabicCharacterTokenizer()

    tokenizer.fit(dataset)

    print(
        f"Vocabulary size: "
        f"{tokenizer.vocabulary_size}"
    )

    print()

    # --------------------------------------------------
    # 4. Train different N-gram models
    # --------------------------------------------------

    models = {}

    for n in [2, 3, 4]:

        model = CharacterNGramModel(
            n=n
        )

        model.fit(
            dataset,
            tokenizer,
        )

        models[n] = model

        print(
            f"{n}-gram model:"
        )

        print(
            f"  Number of contexts: "
            f"{len(model)}"
        )

        print(
            f"  Vocabulary size: "
            f"{len(model.get_vocabulary())}"
        )

        print()

    # --------------------------------------------------
    # 5. Inspect an example context
    # --------------------------------------------------

    trigram_model = models[3]

    context = "ال"

    probabilities = (
        trigram_model
        .get_next_token_probabilities(
            context
        )
    )

    print(
        f"Next-character probabilities "
        f"for context '{context}':"
    )

    # Sort by probability.
    sorted_probabilities = sorted(
        probabilities.items(),
        key=lambda item: item[1],
        reverse=True,
    )

    for token, probability in sorted_probabilities[
        :10
    ]:

        print(
            f"  {repr(token)}: "
            f"{probability:.4f}"
        )

    print()

    # --------------------------------------------------
    # 6. Generate text
    # --------------------------------------------------

    start_prompt = "يوم واحد"

    print(
        f"Start prompt:\n"
        f"{start_prompt}"
    )

    print()

    for n, model in models.items():

        generator = NGramTextGenerator(
            model,
            tokenizer,
        )

        print(
            f"{n}-gram greedy:"
        )

        generated_text = (
            generator.generate(
                start_prompt=start_prompt,
                n_tokens=50,
                sampling_mode="greedy",
            )
        )

        print(generated_text)

        print()

    # --------------------------------------------------
    # 7. Random generation
    # --------------------------------------------------

    model = models[4]

    generator = NGramTextGenerator(
        model,
        tokenizer,
    )

    print(
        "4-gram random generation:"
    )

    generated_text = generator.generate(
        start_prompt=start_prompt,
        n_tokens=50,
        sampling_mode="random",
    )

    print(generated_text)


if __name__ == "__main__":
    main()