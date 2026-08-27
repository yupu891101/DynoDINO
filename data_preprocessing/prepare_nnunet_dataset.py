import argparse
import json
import os
from pathlib import Path

import numpy as np
import SimpleITK as sitk
from tqdm.auto import tqdm

DATASET_CONFIG = {
    "lits": {
        "name": "Dataset003_Liver",
        "source_name": "Task03_Liver",
        "image_dir": "npy_imagesTr",
        "label_dir": "npy_labelsTr",
        "image_prefix": "liver",
        "num_channels": 1,
        "channel_patterns": [None],
        "channel_names": {
            "0": "ct",
        },
        "patient_id_fn": lambda name: name.split("_")[1],
        "is_test_fn": lambda name, test_ids: name in test_ids,
    },
    "cect": {
        "name": "Dataset500_CECT",
        "source_name": "CECT",
        "image_dir": "ct_npy_files",
        "label_dir": "label_npy_files",
        "image_prefix": "CECT",
        "num_channels": 3,
        "channel_patterns": [
            "00_",
            "01_",
            "02_",
        ],
        "channel_names": {
            "1": "arterial",
            "2": "portal venous",
            "3": "delayed",
        },
        "patient_id_fn": lambda name: name[-3:],
        "is_test_fn": lambda name, test_ids: name in test_ids,
    },
    "wawtace": {
        "name": "Dataset501_WAWTACE",
        "source_name": "WAWTACE",
        "image_dir": "ct_npy_files",
        "label_dir": "label_npy_files",
        "image_prefix": "TACE",
        "num_channels": 3,
        "channel_patterns": [
            "00_",
            "01_",
            "02_",
        ],
        "channel_names": {
            "1": "arterial",
            "2": "portal venous",
            "3": "delayed",
        },
        "patient_id_fn": lambda name: name,
        "is_test_fn": lambda name, test_ids: name in test_ids,
    },
}


def remove_macos_files(data_root: Path) -> None:
    """Remove macOS metadata files such as ._* recursively."""
    removed = 0

    for file in data_root.glob("**/._*"):
        if file.is_file():
            file.unlink()
            removed += 1

    if removed > 0:
        print(f"Removed {removed} macOS metadata files.")


def load_test_ids(fold_list_path: Path) -> set[str]:
    """Load test-set patient IDs from fold_list.json."""
    with fold_list_path.open(encoding="utf-8-sig") as f:
        dataset_info = json.load(f)

    if not dataset_info:
        raise ValueError(f"Empty fold list: {fold_list_path}")

    test_ids = set(dataset_info[0].keys())

    print(f"Loaded {len(test_ids)} test IDs from {fold_list_path}")

    return test_ids


def npy_folder_to_volume(
    folder: Path,
    channel_pattern: str | None = None,
) -> np.ndarray:
    """Load NPY slices and stack them into a 3D volume."""

    if channel_pattern is None:
        slice_files = sorted(folder.glob("*.npy"))
    else:
        slice_files = sorted(
            file
            for file in folder.glob("*.npy")
            if channel_pattern in file.name
        )

    if not slice_files:
        pattern_info = (
            "*.npy"
            if channel_pattern is None
            else f"*{channel_pattern}*.npy"
        )

        raise FileNotFoundError(
            f"No NPY slices found with pattern '{pattern_info}' "
            f"in {folder}"
        )

    slices = [np.load(file) for file in slice_files]

    return np.stack(slices, axis=0)


def save_volume(
    volume: np.ndarray,
    output_path: Path,
) -> None:
    """Convert a NumPy volume to NIfTI and save it."""
    image = sitk.GetImageFromArray(volume)
    sitk.WriteImage(image, output_path)


def convert_images(
    source_dir: Path,
    output_train_dir: Path,
    output_test_dir: Path,
    test_ids: set[str],
    config: dict,
) -> None:
    """Convert preprocessed image NPY files into nnUNet image files."""

    folders = sorted(
        folder
        for folder in source_dir.iterdir()
        if folder.is_dir()
    )

    for folder in tqdm(
        folders,
        desc=f"Converting {config['source_name']} images",
    ):
        folder_name = folder.name
        patient_id = config["patient_id_fn"](folder_name)
        patient_id = patient_id.zfill(3)

        is_test = config["is_test_fn"](folder_name, test_ids)

        output_dir = (
            output_test_dir
            if is_test
            else output_train_dir
        )

        for channel_idx, channel_pattern in enumerate(
            config["channel_patterns"]
        ):
            volume = npy_folder_to_volume(
                folder,
                channel_pattern,
            )

            channel_number = (
                channel_idx
                if config["num_channels"] == 1
                else channel_idx + 1
            )

            output_path = output_dir / (
                f"{config['image_prefix']}_"
                f"{patient_id}_"
                f"{channel_number:04d}.nii.gz"
            )

            save_volume(volume, output_path)


def convert_labels(
    source_dir: Path,
    output_train_dir: Path,
    output_test_dir: Path,
    test_ids: set[str],
    config: dict,
) -> None:
    """Convert preprocessed label NPY files into nnUNet label files."""

    folders = sorted(
        folder
        for folder in source_dir.iterdir()
        if folder.is_dir()
    )

    for folder in tqdm(
        folders,
        desc=f"Converting {config['source_name']} labels",
    ):
        folder_name = folder.name
        patient_id = config["patient_id_fn"](folder_name)
        patient_id = patient_id.zfill(3)

        is_test = config["is_test_fn"](folder_name, test_ids)

        output_dir = (
            output_test_dir
            if is_test
            else output_train_dir
        )

        volume = npy_folder_to_volume(
            folder,
            config["channel_patterns"][0],
        )

        output_path = output_dir / (
            f"{config['image_prefix']}_"
            f"{patient_id}.nii.gz"
        )

        save_volume(volume, output_path)


def create_dataset_json(
    dataset_root: Path,
    config: dict,
) -> None:
    """Create the nnUNet dataset.json file."""

    labels_tr_dir = dataset_root / "labelsTr"
    images_tr_dir = dataset_root / "imagesTr"

    count_training = len(
        list(labels_tr_dir.glob("*.nii.gz"))
    )

    image_files = sorted(
        images_tr_dir.glob("*.nii.gz")
    )

    if not image_files:
        raise FileNotFoundError(
            f"No training images found in {images_tr_dir}"
        )

    channel_info = config["channel_names"]

    dataset_info = {
        "channel_names": channel_info,
        "labels": {
            "background": 0,
            "liver": 1,
            "cancer": 2,
        },
        "numTraining": count_training,
        "file_ending": ".nii.gz",
    }

    output_path = dataset_root / "dataset.json"

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(
            dataset_info,
            f,
            indent=4,
            ensure_ascii=False,
        )

    print(f"Created: {output_path}")
    print(f"Training cases: {count_training}")
    print(f"Channels: {channel_info}")


def prepare_output_dirs(dataset_root: Path) -> None:
    """Create nnUNet directory structure."""

    for directory in [
        dataset_root / "imagesTr",
        dataset_root / "imagesTs",
        dataset_root / "labelsTr",
        dataset_root / "labelsTs",
    ]:
        directory.mkdir(
            parents=True,
            exist_ok=True,
        )


def convert_dataset(
    dataset_key: str,
    preprocessed_data_root: Path,
    nnunet_raw_root: Path,
) -> None:
    """Run the complete conversion pipeline for one dataset."""

    if dataset_key not in DATASET_CONFIG:
        raise ValueError(
            f"Unknown dataset: {dataset_key}. "
            f"Available: {list(DATASET_CONFIG)}"
        )

    config = DATASET_CONFIG[dataset_key]

    source_root = (
        preprocessed_data_root / config["source_name"]
    )

    dataset_root = (
        nnunet_raw_root / config["name"]
    )

    print("=" * 70)
    print(f"Dataset: {dataset_key}")
    print(f"Source:  {source_root}")
    print(f"Output:  {dataset_root}")
    print("=" * 70)

    if not source_root.exists():
        raise FileNotFoundError(
            f"Source dataset does not exist: {source_root}"
        )

    fold_list_path = source_root / "fold_list.json"

    if not fold_list_path.exists():
        raise FileNotFoundError(
            f"fold_list.json not found: {fold_list_path}"
        )

    remove_macos_files(source_root)

    test_ids = load_test_ids(fold_list_path)

    prepare_output_dirs(dataset_root)

    image_source = (
        source_root / config["image_dir"]
    )

    label_source = (
        source_root / config["label_dir"]
    )

    convert_images(
        source_dir=image_source,
        output_train_dir=dataset_root / "imagesTr",
        output_test_dir=dataset_root / "imagesTs",
        test_ids=test_ids,
        config=config,
    )

    convert_labels(
        source_dir=label_source,
        output_train_dir=dataset_root / "labelsTr",
        output_test_dir=dataset_root / "labelsTs",
        test_ids=test_ids,
        config=config,
    )

    create_dataset_json(
        dataset_root=dataset_root,
        config=config,
    )

    print()
    print(f"Finished: {dataset_key}")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Convert DynoDINO preprocessed NPY datasets "
            "into nnUNet-compatible NIfTI datasets."
        )
    )

    parser.add_argument(
        "--dataset",
        required=True,
        choices=list(DATASET_CONFIG.keys()) + ["all"],
        help="Dataset to convert.",
    )

    parser.add_argument(
        "--preprocessed-data-root",
        type=Path,
        default=Path("../DynoDINO_preprocessed_data"),
        help="Root directory containing preprocessed datasets.",
    )

    parser.add_argument(
        "--nnunet-raw-root",
        type=Path,
        default=None,
        help="Root directory for nnUNet raw datasets.",
    )

    args = parser.parse_args()

    if args.nnunet_raw_root is None:
        nnunet_raw = os.environ.get("nnUNet_raw")

        if nnunet_raw is None:
            parser.error(
                "nnUNet raw directory is not specified. "
                "Please set the nnUNet_raw environment variable "
                "or provide --nnunet-raw-root."
            )

        args.nnunet_raw_root = Path(nnunet_raw)

    if args.dataset == "all":
        for dataset_key in DATASET_CONFIG:
            convert_dataset(
                dataset_key=dataset_key,
                preprocessed_data_root=args.preprocessed_data_root,
                nnunet_raw_root=args.nnunet_raw_root,
            )
    else:
        convert_dataset(
            dataset_key=args.dataset,
            preprocessed_data_root=args.preprocessed_data_root,
            nnunet_raw_root=args.nnunet_raw_root,
        )


if __name__ == "__main__":
    main()