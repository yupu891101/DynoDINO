import json
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import tv_tensors
from torchvision.transforms import v2

from utils.file_utils import extract_data_file, folder_validate
from utils.utils import chunk_list, file_downloader

from .data_reader import msd_base_data_reader

TaskNames = Literal[
    "Task01_BrainTumour",
    "Task02_Heart",
    "Task03_Liver",
    "Task04_Hippocampus",
    "Task05_Prostate",
    "Task06_Lung",
    "Task07_Pancreas",
    "Task08_HepaticVessel",
    "Task09_Spleen",
    "Task10_Colon",
]

TASK_DATA_URLS: dict[TaskNames | str, str] = {
    "Task01_BrainTumour": "https://msd-for-monai.s3-us-west-2.amazonaws.com/Task01_BrainTumour.tar",
    "Task02_Heart": "https://msd-for-monai.s3-us-west-2.amazonaws.com/Task02_Heart.tar",
    "Task03_Liver": "https://msd-for-monai.s3-us-west-2.amazonaws.com/Task03_Liver.tar",
    "Task04_Hippocampus": "https://msd-for-monai.s3-us-west-2.amazonaws.com/Task04_Hippocampus.tar",
    "Task05_Prostate": "https://msd-for-monai.s3-us-west-2.amazonaws.com/Task05_Prostate.tar",
    "Task06_Lung": "https://msd-for-monai.s3-us-west-2.amazonaws.com/Task06_Lung.tar",
    "Task07_Pancreas": "https://msd-for-monai.s3-us-west-2.amazonaws.com/Task07_Pancreas.tar",
    "Task08_HepaticVessel": "https://msd-for-monai.s3-us-west-2.amazonaws.com/Task08_HepaticVessel.tar",
    "Task09_Spleen": "https://msd-for-monai.s3-us-west-2.amazonaws.com/Task09_Spleen.tar",
    "Task10_Colon": "https://msd-for-monai.s3-us-west-2.amazonaws.com/Task10_Colon.tar",
}
IMAGE_NPY_FOLDER = "npy_imagesTr"
LABEL_NPY_FOLDER = "npy_labelsTr"
FOLD_LIST_NAME = "fold_list.json"


class MSDBaseDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(
        self,
        data_root: str,
        task_name: TaskNames | str,
        num_workers: int = 4,
        split: Literal["train", "test"] = "train",
        fold_num: Literal[5, 10] = 5,
        fold_index: int = 0,
        modality_num: int | None = None,
        transform: v2.Transform | None = None,
    ):
        """MSDBaseDataset will automatically download the data of supported task_name from *MSD Challenge* source.
        MSDBaseDataset also accepts all datasets following the *MSD Challenge* format.
        If you want to prepare your custom data, you can follow the file structure below:
        ```
        <data_root>/
        └── <task_name>/
            ├── imagesTr/
            │   ├── XXX_<id>.nii.gz (or XXX_<modal_id>_<id>.nii.gz if is a multi modal data)
            │   ...
            ├── labelsTr/
            │   ├── XXX_<id>.nii.gz (or XXX_<modal_id>_<id>.nii.gz if is a multi modal data)
            │   ...
            └── dataset.json (Optional)
        ```
        The `dataset.json` file is used to get the number of modalities.
        You can also pass the modality number directly as an input argument.
        If your dataset is multi-modal,
        the file names should contain a modal ID (e.g., `XXX_00_25.nii.gz`, `XXX_01_25.nii.gz`, ...),
        where the modal ID is a 2-digit number starting from "00".

        **Args**:
            - `data_root` | *(str)*: The root directory where the dataset is stored or will be downloaded.
            - `task_name` | *(TaskNames | str)*: The name of the task (e.g., "Task01_BrainTumour").
            - `num_workers` | *(int, optional)*: Number of workers to use for data processing. Defaults to 4.
            - `split` | *(Literal["train", "test"], optional)*:
                The dataset split to use ("train" or "test"). Defaults to "train".
            - `fold_num` | *(Literal[5, 10], optional)*:
                The number of folds for cross-validation (5 or 10). Defaults to 5.
            - `fold_index` | *(int, optional)*: The index of the current fold to use. Defaults to 0.
            - `modality_num` | *(int | None, optional)*:
                The number of modalities in the dataset. If None, it will try to read from `dataset.json`.
            - `transform` | *(Callable[..., torch.Tensor] | None, optional)*:
                A callable transformation to apply to the input images. Defaults to None.
        """
        self.data_root = Path(data_root)
        folder_validate(self.data_root)
        if fold_index > fold_num - 1:
            raise IndexError("fold_index out of range, fold_index should be less than fold_num - 1.")

        if self.data_root.joinpath(task_name).exists():
            # valid task folder exists
            self.data_root = self.data_root.joinpath(task_name)

        elif self.data_root.joinpath(f"{task_name}.tar").exists():
            # valid tar file exists, extract it
            extracted_folder = extract_data_file(self.data_root.joinpath(f"{task_name}.tar").as_posix())
            if len(extracted_folder) > 1:
                raise ValueError(
                    f"Expected only one folder to be extracted for task {task_name}, "
                    f"but found multiple: {extracted_folder}"
                )
            self.data_root = self.data_root.joinpath(extracted_folder[0])
        else:
            # download and extract the dataset
            task_url = TASK_DATA_URLS.get(task_name)
            if not task_url:
                raise ValueError(f"Invalid task name: {task_name}")

            tar_save_path = self.data_root.joinpath(f"{task_name}.tar").as_posix()
            file_downloader(task_url, tar_save_path, log_error=False)
            extracted_folder = extract_data_file(tar_save_path)
            if len(extracted_folder) > 1:
                raise ValueError(
                    f"Expected only one folder to be extracted for task {task_name}, "
                    f"but found multiple: {extracted_folder}"
                )
            self.data_root = self.data_root.joinpath(extracted_folder[0])

        # prepare npy data if not exists
        if (
            not self.data_root.joinpath(IMAGE_NPY_FOLDER).exists()
            or not self.data_root.joinpath(LABEL_NPY_FOLDER).exists()
        ):
            msd_base_data_reader(self.data_root.as_posix(), IMAGE_NPY_FOLDER, LABEL_NPY_FOLDER, num_workers=num_workers)

        self.fold_list = self.gen_fold_list()
        flatten_fold = self.fold_list_flatten(fold_num, fold_index, split)

        self.modality_num = self.get_modality_num(modality_num)
        self.file_list = self.gen_data_list(flatten_fold)
        self.__transforms = None
        self.transforms = transform

    @property
    def transforms(self) -> v2.Transform | None:
        return self.__transforms

    @transforms.setter
    def transforms(self, transforms: v2.Transform | None):
        if not isinstance(transforms, (v2.Transform)):
            if transforms is not None:
                raise TypeError(
                    "The input transforms should be a subclass of torchvision.transform.v2.Transform, "
                    f"got {type(transforms)}"
                )
        self.__transforms = transforms

    def get_modality_num(self, input_modality_num: int | None) -> int:
        dataset_json_path = self.data_root.joinpath("dataset.json")
        if not dataset_json_path.exists():
            if input_modality_num is not None:
                return input_modality_num
            else:
                raise ValueError("modality_num is not set and dataset.json is missing.")

        with open(dataset_json_path, encoding="utf-8") as f:
            dataset_info = json.load(f)

        modalities = dataset_info.get("modality")
        if modalities is None:
            if input_modality_num is not None:
                return input_modality_num
            else:
                raise ValueError("modality information is missing in dataset.json and modality_num is not set.")

        return len(modalities)

    def gen_fold_list(self) -> list[dict[str, list[str]]]:
        if self.data_root.joinpath(FOLD_LIST_NAME).exists():
            with open(self.data_root.joinpath(FOLD_LIST_NAME), encoding="utf-8-sig") as j:
                fold_list = json.load(j)
            return fold_list

        all_image_patient_ids = sorted(self.data_root.joinpath(IMAGE_NPY_FOLDER).glob("*"))
        all_label_patient_ids = sorted(self.data_root.joinpath(LABEL_NPY_FOLDER).glob("*"))

        # last 2 parts of all_image_ids is patient_id/ and slice_id.npy
        mismatch_ids = [
            ("/".join(image_id.parts[-2:]), "/".join(label_id.parts[-2:]))
            for image_id, label_id in zip(all_image_patient_ids, all_label_patient_ids, strict=True)
            if image_id.name != label_id.name
        ]
        if len(mismatch_ids) > 0:
            raise ValueError(f"The data id of image and label should be same, got: {mismatch_ids}")

        fold_list: list[dict[str, list[str]]] = []
        for fold in chunk_list(all_image_patient_ids):
            single_fold_dict = {}
            for folder in fold:
                single_fold_dict[folder.name] = sorted(file.name for file in folder.glob("*.npy"))
            fold_list.append(single_fold_dict)
        with open(self.data_root.joinpath(FOLD_LIST_NAME), "w", encoding="utf-8-sig") as j:
            json.dump(fold_list, j, ensure_ascii=False, indent=4)
        return fold_list

    def fold_list_flatten(self, fold_num: Literal[5, 10], fold_index: int, split: Literal["train", "test"]):
        if fold_num == 5:
            test_indices = [fold_index * 2, fold_index * 2 + 1]
        else:
            test_indices = [fold_index]
        train_indices = [i for i in range(10) if i not in test_indices]
        if split == "train":
            fold_indices = train_indices
        else:
            fold_indices = test_indices
        flatten_fold: list[str] = []
        for idx in fold_indices:
            for image_id, slice_list in self.fold_list[idx].items():
                flatten_fold.extend(f"{image_id}/{slice_id}" for slice_id in slice_list)

        return flatten_fold

    def gen_data_list(self, flatten_fold: list[str]) -> list[dict[str, list[str] | str]]:
        data_list = []
        npy_images_tr_path = self.data_root.joinpath(IMAGE_NPY_FOLDER)
        npy_labels_tr_path = self.data_root.joinpath(LABEL_NPY_FOLDER)
        if self.modality_num == 1:
            file_pairs = zip(
                sorted(npy_images_tr_path.joinpath(file_path) for file_path in flatten_fold),
                sorted(npy_labels_tr_path.joinpath(file_path) for file_path in flatten_fold),
                strict=False,
            )
            entry_generator = mono_modal_generator
        else:
            file_pairs = zip(
                sorted(npy_images_tr_path.joinpath(file_path) for file_path in flatten_fold if "_00_" in file_path),
                sorted(npy_labels_tr_path.joinpath(file_path) for file_path in flatten_fold if "_00_" in file_path),
                strict=False,
            )
            entry_generator = multi_modal_generator

        for image_file, label_file in file_pairs:
            data_entry = entry_generator(image_file, label_file, self.modality_num)
            data_list.append(data_entry)

        return data_list

    def __len__(self):
        return len(self.file_list)

    def load_image(self, image_path: str | list[str]) -> torch.Tensor:
        if isinstance(image_path, str):
            image_tensor = torch.from_numpy(np.load(image_path)).unsqueeze(0)
        else:
            image_slices = [torch.from_numpy(np.load(modality_path)) for modality_path in image_path]
            image_tensor = torch.stack(image_slices, dim=0)

        return image_tensor

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        image_tensor: torch.Tensor
        label_tensor: torch.Tensor
        file_entry = self.file_list[idx]
        image_tensor = self.load_image(file_entry["image"]).float()
        image_tensor = (image_tensor.clip(-1024, 1024) + 1024) / 2048
        label_tensor = tv_tensors.Mask(self.load_image(file_entry["label"]).long())
        if self.transforms is not None:
            image_tensor, label_tensor = self.transforms(image_tensor, label_tensor)

        return image_tensor, label_tensor


def mono_modal_generator(image_path: Path, label_path: Path, modality_num: int) -> dict[str, str]:
    case_id = image_path.stem.split("_")[0]
    return {
        "image": image_path.as_posix(),
        "label": label_path.as_posix(),
        "case_id": case_id,
    }


def multi_modal_generator(image_path: Path, label_path: Path, modality_num: int) -> dict[str, list[str] | str]:
    case_id = image_path.stem.split("_")[0]
    return {
        "image": [
            image_path.as_posix().replace("_00_", f"_{modality_idx:02d}_") for modality_idx in range(modality_num)
        ],
        "label": [
            label_path.as_posix().replace("_00_", f"_{modality_idx:02d}_") for modality_idx in range(modality_num)
        ],
        "case_id": case_id,
    }


def get_transforms(
    transform_type: Literal["resize", "crop", "resize_crop"] = "resize_crop", size: tuple[int, int] = (256, 256)
):
    # --- New MedDINOv3 Style Transforms ---
    spatial_transforms = [
        v2.RandomApply(
            [
                v2.RandomAffine(
                    degrees=(-30.0, 30.0),  # Aligned with dinov3Trainer (-30, 30)
                    scale=(0.7, 1.4),       # Aligned with dinov3Trainer (0.7, 1.4)
                    shear=0,
                    interpolation=v2.InterpolationMode.BILINEAR,
                )
            ],
            p=0.2,  # Aligned with dinov3Trainer p_rotation=0.2
        ),
        v2.RandomHorizontalFlip(p=0.5),
        v2.RandomVerticalFlip(p=0.5),
        # v2.RandomErasing(p=0.1), # Removed to avoid warning and preserve small lesions
        # v2.RandomApply([v2.RandomRotation(degrees=[-45, 45])], p=0.5), # Handled by RandomAffine
        v2.RandomApply([v2.ElasticTransform(alpha=(50.0, 100.0), sigma=5.0)], p=0.1), # Standard nnUNet-like elastic
    ]

    intensity_transforms = [
        # Multiplicative Brightness & Contrast (Simulating dinov3Trainer)
        v2.RandomApply(
            [
                v2.ColorJitter(
                    brightness=(0.75, 1.25),
                    contrast=(0.75, 1.25),
                    saturation=0,
                    hue=0,
                )
            ],
            p=0.15,
        ),
        v2.RandomApply([v2.GaussianNoise(mean=0.0, sigma=0.1)], p=0.1),
        v2.RandomApply([v2.GaussianBlur(kernel_size=3, sigma=(0.5, 1.0))], p=0.2),
        v2.RandomAutocontrast(p=0.1),
        # v2.RandomEqualize(p=0.2), # Often too aggressive for CT
        # v2.RandomInvert(p=0.1),   # Medical CT images should not be inverted
    ]

    # --- Original Code (Commented Out) ---
    """
    spatial_transforms = [
        v2.RandomApply(
            [
                v2.RandomAffine(  # 複合幾何變換
                    degrees=(-30.0, 30.0),
                    scale=(0.7, 1.4),
                    shear=0,
                    interpolation=v2.InterpolationMode.BILINEAR,
                )
            ],
            p=0.5,  # 執行 RandomAffine 的機率
        ),
        # 鏡像 (Flip) 接受 'p' 參數，不需要 RandomApply
        v2.RandomHorizontalFlip(p=0.5),
        v2.RandomVerticalFlip(p=0.5),
        v2.RandomErasing(p=0.1),
        v2.RandomApply([v2.RandomRotation(degrees=[-45, 45])], p=0.5),
        v2.RandomApply([v2.ElasticTransform()], p=0.1),
    ]
    intensity_transforms = [
        # 亮度/對比度
        v2.RandomApply(
            [
                v2.ColorJitter(
                    brightness=(0.75, 1.25),
                    contrast=(0.75, 1.25),
                    saturation=0,
                    hue=0,
                )
            ],
            p=0.15,
        ),
        # 雜訊
        v2.RandomApply([v2.GaussianNoise(mean=0.0, sigma=0.1)], p=0.1),
        # 模糊
        v2.RandomApply([v2.GaussianBlur(kernel_size=3, sigma=(0.5, 1.0))], p=0.2),
        # Gamma 校正/對比度調整
        v2.RandomAutocontrast(p=0.1),
        v2.RandomEqualize(p=0.2),
        v2.RandomInvert(p=0.1),
    ]
    """
    final_transforms = []
    if transform_type == "crop":
        final_transforms.append(v2.RandomCrop(size))
    elif transform_type == "resize":
        final_transforms.append(v2.Resize(size))
    else:
        final_transforms.append(v2.RandomResizedCrop(size))
    all_transforms = [*spatial_transforms, *intensity_transforms, *final_transforms]
    return v2.Compose(all_transforms)


def validation_transforms(transform_type: Literal["crop", "resize"] = "resize", size: tuple[int, int] = (256, 256)):
    deterministic_transforms = []
    if transform_type == "crop":
        # return v2.RandomCrop(size)
        deterministic_transforms.append(v2.RandomCrop(size))
    else:
        # return v2.Resize(size)
        deterministic_transforms.append(v2.Resize(size, interpolation=v2.InterpolationMode.BILINEAR))
    return v2.Compose(deterministic_transforms)
