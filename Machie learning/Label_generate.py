import argparse
import re
from pathlib import Path

import pandas as pd


CONCENTRATION_MAP = {
    "0": 0.0,
    "1": 0.165,
    "3": 0.33,
}


def extract_concentration(class_name: str, component_code: str) -> float:
    """
    Extract concentration from the class folder name.

    Expected examples:
        E1 -> Ethanol = 0.165
        A3 -> Acetone = 0.33
        I0 -> IPA = 0.0

    Parameters
    ----------
    class_name : str
        Name of the class folder.
    component_code : str
        Component identifier: "E" for ethanol, "A" for acetone, "I" for IPA.

    Returns
    -------
    float
        Concentration value.
    """
    match = re.search(rf"{component_code}([013])", class_name, re.IGNORECASE)

    if match is None:
        return 0.0

    level = match.group(1)
    return CONCENTRATION_MAP.get(level, 0.0)


def create_label_csv(dataset_dir: Path, output_csv_path: Path) -> None:
    """
    Create a label CSV file from a dataset organized by class folders.

    The output CSV does not include local absolute paths to avoid exposing
    private directory information.
    """
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")

    class_dirs = sorted(
        [
            item
            for item in dataset_dir.iterdir()
            if item.is_dir()
        ]
    )

    records = []

    for class_index, class_dir in enumerate(class_dirs):
        class_name = class_dir.name

        ethanol = extract_concentration(class_name, "E")
        acetone = extract_concentration(class_name, "A")
        ipa = extract_concentration(class_name, "I")

        csv_files = sorted(class_dir.glob("*.csv"))

        for sample_index, csv_file in enumerate(csv_files):
            sample_id = f"class_{class_index:02d}_sample_{sample_index:03d}"

            records.append(
                {
                    "sample_id": sample_id,
                    "filename": csv_file.name,
                    "label": class_index,
                    "ethanol": ethanol,
                    "acetone": acetone,
                    "ipa": ipa,
                }
            )

    df = pd.DataFrame(
        records,
        columns=[
            "sample_id",
            "filename",
            "label",
            "ethanol",
            "acetone",
            "ipa",
        ],
    )

    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv_path, index=False, encoding="utf-8-sig")

    print(f"Label file created successfully.")
    print(f"Number of samples: {len(df)}")
    print(f"Output file: {output_csv_path}")
    print("\nPreview:")
    print(df.head())


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create a label CSV file for the photonic in-sensor computing dataset."
    )

    parser.add_argument(
        "--dataset_dir",
        type=Path,
        required=True,
        help="Path to the dataset directory containing class folders.",
    )

    parser.add_argument(
        "--output_csv",
        type=Path,
        default=Path("EAI_label.csv"),
        help="Path to save the output label CSV file.",
    )

    args = parser.parse_args()

    create_label_csv(
        dataset_dir=args.dataset_dir,
        output_csv_path=args.output_csv,
    )


if __name__ == "__main__":
    main()
