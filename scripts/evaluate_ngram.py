"""
Train and evaluate Bigram, Trigram, and 4-gram models.

This script performs:

    1. Dataset loading
    2. Dataset cleaning
    3. Train/test split
    4. Tokenizer construction
    5. N-gram training
    6. Cross entropy calculation
    7. Perplexity calculation
    8. Model comparison
    9. Visualization
    10. Saving evaluation results
"""

from __future__ import annotations

import csv
import random
from pathlib import Path
import sys

# PROJECT PATH

PROJECT_ROOT = (
    Path(__file__).resolve().parents[1]
)

if str(PROJECT_ROOT) not in sys.path:

    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )

# PROJECT IMPORTS

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

from src.evaluation import (
    evaluate_model,
    calculate_unknown_context_rate,
)

# CONFIGURATION

RANDOM_SEED = 42

TEST_SIZE = 0.20

NGRAM_SIZES = [2, 3, 4]

DATASET_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "arabic_sample.txt"
)

RESULTS_DIR = (
    PROJECT_ROOT
    / "results"
    / "week3"
)

# TRAIN / TEST SPLIT

def train_test_split(
    dataset: list[str],
    test_size: float = 0.2,
    seed: int = 42,
) -> tuple[list[str], list[str]]:
    """
    Split a dataset into training and testing sets.

    Args:
        dataset:
            List of text samples.

        test_size:
            Fraction of data used for testing.

        seed:
            Random seed for reproducibility.

    Returns:
        Tuple of:

            train_dataset
            test_dataset
    """

    if not dataset:

        raise ValueError(
            "Dataset is empty."
        )

    if not 0 < test_size < 1:

        raise ValueError(
            "test_size must be between 0 and 1."
        )

    shuffled_dataset = list(
        dataset
    )

    random.Random(seed).shuffle(
        shuffled_dataset
    )

    test_count = max(
        1,
        int(
            len(shuffled_dataset)
            * test_size
        ),
    )

    test_dataset = (
        shuffled_dataset[
            :test_count
        ]
    )

    train_dataset = (
        shuffled_dataset[
            test_count:
        ]
    )

    if not train_dataset:

        raise ValueError(
            "Training dataset became empty. "
            "Use a larger dataset."
        )

    return (
        train_dataset,
        test_dataset,
    )

# SAVE RESULTS

def save_results(
    results: list[dict],
    output_path: Path,
) -> None:
    """
    Save evaluation results to CSV.
    """

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fieldnames = [
        "model",
        "n",
        "contexts",
        "vocabulary_size",
        "cross_entropy",
        "perplexity",
        "unknown_context_rate",
    ]

    with output_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        for result in results:

            writer.writerow(
                result
            )

# VISUALIZATION

def create_visualization(
    results: list[dict],
    output_path: Path,
) -> None:
    """
    Create a model comparison visualization.

    The figure contains:

        - Cross entropy comparison
        - Perplexity comparison
    """

    import matplotlib.pyplot as plt

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    models = [
        result["model"]
        for result in results
    ]

    cross_entropy = [
        result["cross_entropy"]
        for result in results
    ]

    perplexity = [
        result["perplexity"]
        for result in results
    ]

    # Cross entropy

    plt.figure(
        figsize=(8, 5)
    )

    plt.bar(
        models,
        cross_entropy,
    )

    plt.xlabel(
        "Model"
    )

    plt.ylabel(
        "Cross entropy (bits)"
    )

    plt.title(
        "Character-Level N-gram Cross Entropy"
    )

    plt.tight_layout()

    cross_entropy_path = (
        output_path.parent
        / "cross_entropy_comparison.png"
    )

    plt.savefig(
        cross_entropy_path,
        dpi=200,
    )

    plt.close()

    # Perplexity

    plt.figure(
        figsize=(8, 5)
    )

    plt.bar(
        models,
        perplexity,
    )

    plt.xlabel(
        "Model"
    )

    plt.ylabel(
        "Perplexity"
    )

    plt.title(
        "Character-Level N-gram Perplexity"
    )

    plt.tight_layout()

    perplexity_path = (
        output_path.parent
        / "perplexity_comparison.png"
    )

    plt.savefig(
        perplexity_path,
        dpi=200,
    )

    plt.close()

# MAIN

def main() -> None:
    """
    Run the complete Week 3 evaluation pipeline.
    """

    print("=" * 60)

    print(
        "WEEK 3 — N-GRAM EVALUATION"
    )

    print("=" * 60)

    # 1. Load dataset

    print(
        "\n[1] Loading dataset..."
    )

    dataset = load_text_file(
        DATASET_PATH
    )

    dataset = clean_dataset(
        dataset
    )

    print(
        f"Total samples: {len(dataset)}"
    )

    # 2. Train / test split

    print(
        "\n[2] Creating train/test split..."
    )

    train_dataset, test_dataset = (
        train_test_split(
            dataset,
            test_size=TEST_SIZE,
            seed=RANDOM_SEED,
        )
    )

    print(
        f"Training samples: "
        f"{len(train_dataset)}"
    )

    print(
        f"Testing samples: "
        f"{len(test_dataset)}"
    )

    # 3. Build tokenizer

    print(
        "\n[3] Building tokenizer..."
    )

    tokenizer = (
        ArabicCharacterTokenizer()
    )

    # IMPORTANT:
    # The tokenizer vocabulary is learned
    # from TRAINING data only.

    tokenizer.fit(
        train_dataset
    )

    print(
        f"Vocabulary size: "
        f"{tokenizer.vocabulary_size}"
    )

    # 4. Train and evaluate models

    print(
        "\n[4] Training models..."
    )

    results = []

    for n in NGRAM_SIZES:

        print(
            f"\nTraining {n}-gram..."
        )

        model = CharacterNGramModel(
            n=n
        )

        model.fit(
            train_dataset,
            tokenizer,
        )

        # Evaluate ONLY on test data.
        metrics = evaluate_model(
            model,
            test_dataset,
            tokenizer,
        )

        unknown_context_rate = (
            calculate_unknown_context_rate(
                model,
                test_dataset,
                tokenizer,
            )
        )

        result = {
            "model": f"{n}-gram",
            "n": n,
            "contexts": len(model),
            "vocabulary_size": len(
                model.get_vocabulary()
            ),
            "cross_entropy": metrics[
                "cross_entropy"
            ],
            "perplexity": metrics[
                "perplexity"
            ],
            "unknown_context_rate": (
                unknown_context_rate
            ),
        }

        results.append(
            result
        )

        print(
            f"Cross entropy: "
            f"{metrics['cross_entropy']:.4f}"
        )

        print(
            f"Perplexity: "
            f"{metrics['perplexity']:.4f}"
        )

        print(
            f"Unknown context rate: "
            f"{unknown_context_rate:.2%}"
        )

    # 5. Print comparison table

    print(
        "\n[5] Model comparison"
    )

    print()

    print(
        f"{'Model':<10}"
        f"{'Cross Entropy':<18}"
        f"{'Perplexity':<15}"
        f"{'Unknown Context':<18}"
    )

    print(
        "-" * 61
    )

    for result in results:

        print(
            f"{result['model']:<10}"
            f"{result['cross_entropy']:<18.4f}"
            f"{result['perplexity']:<15.4f}"
            f"{result['unknown_context_rate']:<18.2%}"
        )

    # 6. Save CSV

    print(
        "\n[6] Saving results..."
    )

    csv_path = (
        RESULTS_DIR
        / "metrics.csv"
    )

    save_results(
        results,
        csv_path,
    )

    print(
        f"Saved to: {csv_path}"
    )

    # 7. Create visualizations

    print(
        "\n[7] Creating visualizations..."
    )

    visualization_path = (
        RESULTS_DIR
        / "model_comparison.png"
    )

    create_visualization(
        results,
        visualization_path,
    )

    print(
        "Visualization saved."
    )

    # 8. Find best model

    best_model = min(
        results,
        key=lambda result: result[
            "perplexity"
        ],
    )

    print(
        "\n[8] Best model"
    )

    print(
        f"Model: "
        f"{best_model['model']}"
    )

    print(
        f"Cross entropy: "
        f"{best_model['cross_entropy']:.4f}"
    )

    print(
        f"Perplexity: "
        f"{best_model['perplexity']:.4f}"
    )

    print(
        "\nEvaluation complete."
    )


if __name__ == "__main__":
    main()