"""
Prepare the Arabic dataset for the language model.
"""

from pathlib import Path
import sys


# Add project root to Python path.
PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.preprocessing import prepare_dataset


def main() -> None:
    """
    Load the raw dataset, clean it, and save the processed dataset.
    """

    input_path = (
        PROJECT_ROOT
        / "data"
        / "raw"
        / "arabic_sample.txt"
    )

    output_path = (
        PROJECT_ROOT
        / "data"
        / "processed"
        / "cleaned_arabic.txt"
    )

    dataset = prepare_dataset(
        input_path,
        output_path,
    )

    print("Dataset preparation completed.")
    print(f"Input file : {input_path}")
    print(f"Output file: {output_path}")
    print(f"Number of texts: {len(dataset)}")


if __name__ == "__main__":
    main()