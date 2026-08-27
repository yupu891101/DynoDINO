from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


DEFAULT_PREPROCESSED_DATA_ROOT = Path(
    "../DynoDINO_preprocessed_data"
)

DEFAULT_TRAINER = "dynodino_base_primus_multiscale_Trainer"


DATASET_CONFIG = {
    "lits": {
        "dataset_id": 3,
        "module": "data_preprocessing.msd_base_dataset",
        "class_name": "MSDBaseDataset",
        "kwargs": {
            "task_name": "Task03_Liver",
        },
    },
    "cect": {
        "dataset_id": 500,
        "module": "data_preprocessing.cect_dataset",
        "class_name": "CECTDataset",
        "kwargs": {
            "alignment": False,
        },
    },
    "wawtace": {
        "dataset_id": 501,
        "module": "data_preprocessing.waw_tace_dataset",
        "class_name": "WAWTACEDataset",
        "kwargs": {
            "alignment": False,
        },
    },
}


def run_command(command: list[str]) -> None:
    """Run a command and raise an error if it fails."""

    print()
    print("=" * 70)
    print("Running:")
    print(" ".join(str(item) for item in command))
    print("=" * 70)
    print()

    subprocess.run(command, check=True)


def get_dataset_config(dataset_type: str) -> dict:
    """Return configuration for a supported dataset."""

    if dataset_type not in DATASET_CONFIG:
        raise ValueError(
            f"Unknown dataset: {dataset_type}. "
            f"Available datasets: {list(DATASET_CONFIG)}"
        )

    return DATASET_CONFIG[dataset_type]


def get_dataset_id(dataset_type: str) -> int:
    """Return the nnU-Net dataset ID for a dataset type."""

    return get_dataset_config(dataset_type)["dataset_id"]


def prepare_dataset(
    dataset_type: str,
    data_root: Path,
) -> None:
    """Download and preprocess a dataset."""

    config = get_dataset_config(dataset_type)

    module = config["module"]
    class_name = config["class_name"]

    kwargs = dict(config["kwargs"])
    kwargs["data_root"] = data_root

    kwargs_string = ", ".join(
        f"{key}={value!r}"
        for key, value in kwargs.items()
    )

    code = (
        f"from {module} import {class_name}; "
        f"{class_name}({kwargs_string})"
    )

    run_command(
        [
            sys.executable,
            "-c",
            code,
        ]
    )


def prepare_nnunet_dataset(
    dataset_type: str,
    preprocessed_data_root: Path,
    nnunet_raw_root: Path | None,
) -> None:
    """Convert preprocessed data into nnU-Net format."""

    command = [
        sys.executable,
        "data_preprocessing/prepare_nnunet_dataset.py",
        "--dataset",
        dataset_type,
        "--preprocessed-data-root",
        str(preprocessed_data_root),
    ]

    if nnunet_raw_root is not None:
        command.extend(
            [
                "--nnunet-raw-root",
                str(nnunet_raw_root),
            ]
        )

    run_command(command)


def run_nnunet_preprocessing(
    dataset_id: int,
) -> None:
    """Run nnU-Net planning and preprocessing."""

    run_command(
        [
            "nnUNetv2_plan_and_preprocess",
            "-d",
            str(dataset_id),
            "--verify_dataset_integrity",
        ]
    )


def train(
    dataset_id: int,
    trainer: str,
) -> None:
    """Run nnU-Net training with the selected trainer."""

    run_command(
        [
            "nnUNetv2_train",
            str(dataset_id),
            "2d",
            "all",
            "--npz",
            "-tr",
            trainer,
        ]
    )


def run_prepare(args: argparse.Namespace) -> None:
    """Run dataset download and preprocessing."""

    prepare_dataset(
        dataset_type=args.type,
        data_root=args.data_root,
    )


def run_dataset(args: argparse.Namespace) -> None:
    """Prepare the nnU-Net raw dataset."""

    prepare_nnunet_dataset(
        dataset_type=args.type,
        preprocessed_data_root=args.preprocessed_data_root,
        nnunet_raw_root=args.nnunet_raw_root,
    )


def run_preprocess(args: argparse.Namespace) -> None:
    """Run nnU-Net planning and preprocessing."""

    run_nnunet_preprocessing(
        dataset_id=args.dataset_id,
    )


def run_train(args: argparse.Namespace) -> None:
    """Run nnU-Net training."""

    train(
        dataset_id=args.dataset_id,
        trainer=args.trainer,
    )


def run_all(args: argparse.Namespace) -> None:
    """Run the complete DynoDINO pipeline."""

    dataset_id = get_dataset_id(args.type)

    print()
    print("=" * 70)
    print("DynoDINO pipeline")
    print(f"Dataset type: {args.type}")
    print(f"nnU-Net dataset ID: {dataset_id}")
    print(f"Trainer: {args.trainer}")
    print("=" * 70)

    # Step 1: Download and preprocess the source dataset.
    prepare_dataset(
        dataset_type=args.type,
        data_root=args.data_root,
    )

    # Step 2: Convert the processed data into nnU-Net format.
    prepare_nnunet_dataset(
        dataset_type=args.type,
        preprocessed_data_root=args.preprocessed_data_root,
        nnunet_raw_root=args.nnunet_raw_root,
    )

    # Step 3: Run nnU-Net planning and preprocessing.
    run_nnunet_preprocessing(
        dataset_id=dataset_id,
    )

    # Step 4: Train DynoDINO.
    train(
        dataset_id=dataset_id,
        trainer=args.trainer,
    )


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line argument parser."""

    parser = argparse.ArgumentParser(
        description=(
            "DynoDINO data preparation, nnU-Net preprocessing, "
            "and training pipeline."
        )
    )

    subparsers = parser.add_subparsers(
        dest="command",
        required=True,
    )

    # ------------------------------------------------------------------
    # prepare
    # ------------------------------------------------------------------
    prepare_parser = subparsers.add_parser(
        "prepare",
        help="Download and preprocess a dataset.",
    )

    prepare_parser.add_argument(
        "--type",
        required=True,
        choices=list(DATASET_CONFIG),
        help="Dataset to prepare.",
    )

    prepare_parser.add_argument(
        "--data-root",
        type=Path,
        default=DEFAULT_PREPROCESSED_DATA_ROOT,
        help=(
            "Root directory used for dataset download and "
            "preprocessing."
        ),
    )

    prepare_parser.set_defaults(func=run_prepare)

    # ------------------------------------------------------------------
    # dataset
    # ------------------------------------------------------------------
    dataset_parser = subparsers.add_parser(
        "dataset",
        help="Convert preprocessed data into nnU-Net format.",
    )

    dataset_parser.add_argument(
        "--type",
        required=True,
        choices=list(DATASET_CONFIG),
        help="Dataset to convert.",
    )

    dataset_parser.add_argument(
        "--preprocessed-data-root",
        type=Path,
        default=DEFAULT_PREPROCESSED_DATA_ROOT,
        help=(
            "Root directory containing preprocessed datasets."
        ),
    )

    dataset_parser.add_argument(
        "--nnunet-raw-root",
        type=Path,
        default=None,
        help=(
            "Root directory for nnU-Net raw datasets. "
            "If omitted, the nnUNet_raw environment variable "
            "is used."
        ),
    )

    dataset_parser.set_defaults(func=run_dataset)

    # ------------------------------------------------------------------
    # preprocess
    # ------------------------------------------------------------------
    preprocess_parser = subparsers.add_parser(
        "preprocess",
        help="Run nnU-Net planning and preprocessing.",
    )

    preprocess_parser.add_argument(
        "--dataset-id",
        type=int,
        required=True,
        help="nnU-Net dataset ID to preprocess.",
    )

    preprocess_parser.set_defaults(func=run_preprocess)

    # ------------------------------------------------------------------
    # train
    # ------------------------------------------------------------------
    train_parser = subparsers.add_parser(
        "train",
        help="Train using an nnU-Net trainer.",
    )

    train_parser.add_argument(
        "--dataset-id",
        type=int,
        required=True,
        help="nnU-Net dataset ID to train.",
    )

    train_parser.add_argument(
        "--trainer",
        type=str,
        default=DEFAULT_TRAINER,
        help=(
            "nnU-Net trainer class. "
            f"Default: {DEFAULT_TRAINER}"
        ),
    )

    train_parser.set_defaults(func=run_train)

    # ------------------------------------------------------------------
    # all
    # ------------------------------------------------------------------
    all_parser = subparsers.add_parser(
        "all",
        help="Run the complete DynoDINO pipeline.",
    )

    all_parser.add_argument(
        "--type",
        required=True,
        choices=list(DATASET_CONFIG),
        help="Dataset to process.",
    )

    all_parser.add_argument(
        "--data-root",
        type=Path,
        default=DEFAULT_PREPROCESSED_DATA_ROOT,
        help=(
            "Root directory used for dataset download and "
            "preprocessing."
        ),
    )

    all_parser.add_argument(
        "--preprocessed-data-root",
        type=Path,
        default=DEFAULT_PREPROCESSED_DATA_ROOT,
        help=(
            "Root directory containing preprocessed datasets "
            "for nnU-Net conversion."
        ),
    )

    all_parser.add_argument(
        "--nnunet-raw-root",
        type=Path,
        default=None,
        help=(
            "Root directory for nnU-Net raw datasets. "
            "If omitted, the nnUNet_raw environment variable "
            "is used."
        ),
    )

    all_parser.add_argument(
        "--trainer",
        type=str,
        default=DEFAULT_TRAINER,
        help=(
            "nnU-Net trainer class. "
            f"Default: {DEFAULT_TRAINER}"
        ),
    )

    all_parser.set_defaults(func=run_all)

    return parser


def main() -> None:
    """Main entry point."""

    parser = build_parser()
    args = parser.parse_args()

    if hasattr(args, "data_root"):
        args.data_root = (
            args.data_root.expanduser().resolve()
        )

    if hasattr(args, "preprocessed_data_root"):
        args.preprocessed_data_root = (
            args.preprocessed_data_root
            .expanduser()
            .resolve()
        )

    if hasattr(args, "nnunet_raw_root"):
        if args.nnunet_raw_root is not None:
            args.nnunet_raw_root = (
                args.nnunet_raw_root
                .expanduser()
                .resolve()
            )

    args.func(args)


if __name__ == "__main__":
    main()