import csv
import json
import re
from collections import defaultdict
from logging import getLogger
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import tv_tensors
from torchvision.transforms import v2
from tqdm.auto import tqdm

from data_preprocessing.preprocess_lib.alignment_lib.cect_alignment import (
    parallel_processing_pipeline,
)
from utils.file_utils import extract_data_file, folder_validate
from utils.utils import chunk_list, file_downloader

DATA_URL = (
    "https://download.scidb.cn/download?fileId=cfdd5e37e35a013dc0015aa361531b01&path=/V4/CECT.tar&fileName=CECT.tar"
)
META_DATA_URL = "https://download.scidb.cn/download?fileId=e0af91f0f91a3c20b3a8779f7dc03352&path=/V4/patient_data.csv&fileName=patient_data.csv"
CT_NPY_FOLDER = "ct_npy_files"
LABEL_NPY_FOLDER = "label_npy_files"
META_INFO_NAME = "meta_info.csv"
FOLD_LIST_NAME = "fold_list.json"
METADATA_NAME = "patient_data.csv"


class CECTDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(
        self,
        data_root: str,
        num_workers: int = 4,
        split: Literal["train", "test"] = "train",
        fold_num: Literal[5, 10] = 5,
        fold_index: int = 0,
        transform: v2.Transform | None = None,
        alignment: bool = True,
    ):
        """CECTDataset will automatically download the data from *CECT* official source.

        **Args**:
            - `data_root` | *(str)*: The root directory where the dataset is stored or will be downloaded.
            - `num_workers` | *(int, optional)*: Number of workers to use for data processing. Defaults to 4.
            - `split` | *(Literal["train", "test"], optional)*:
                The dataset split to use ("train" or "test"). Defaults to "train".
            - `fold_num` | *(Literal[5, 10], optional)*:
                The number of folds for cross-validation (5 or 10). Defaults to 5.
            - `fold_index` | *(int, optional)*: The index of the current fold to use. Defaults to 0.
            - `transform` | *(Callable[..., torch.Tensor] | None, optional)*:
                A callable transformation to apply to the input images. Defaults to None.
            - `alignment` | *(bool, optional)*: Whether to align the images. Defaults to True.
        """
        self.logger = getLogger("CECTDataset")
        self.dataset_name = "CECT"
        self.data_root = Path(data_root)
        self.alignment = alignment

        # Determine folder names based on alignment
        self.ct_folder = CT_NPY_FOLDER if alignment else "unaligned_ct_npy_files"
        self.label_folder = LABEL_NPY_FOLDER if alignment else "unaligned_label_npy_files"
        self.meta_info_name = META_INFO_NAME if alignment else "unaligned_meta_info.csv"
        self.fold_list_name = FOLD_LIST_NAME if alignment else "unaligned_fold_list.json"

        folder_validate(self.data_root)
        if fold_index > fold_num - 1:
            raise IndexError("fold_index out of range, fold_index should be less than fold_num - 1.")

        if self.data_root.joinpath(self.dataset_name).exists():
            # If the data folder already exist, do nothing
            pass
        elif self.data_root.joinpath(f"{self.dataset_name}.tar").exists():
            # valid tar file exists, extract it
            extracted_folder = extract_data_file(
                self.data_root.joinpath(f"{self.dataset_name}.tar").as_posix(), mkdir=True
            )
            if len(extracted_folder) != 3:
                raise ValueError(
                    f"Expected 3 folder to be extracted for {self.dataset_name} dataset, "
                    f"but found multiple: {extracted_folder}"
                )
        else:
            # download and extract the dataset
            tar_save_path = self.data_root.joinpath(f"{self.dataset_name}.tar").as_posix()
            file_downloader(DATA_URL, tar_save_path, log_error=False)
            extracted_folder = extract_data_file(tar_save_path)
            if len(extracted_folder) != 1:
                raise ValueError(
                    f"Expected 3 folder to be extracted for {self.dataset_name} dataset, "
                    f"but found multiple: {extracted_folder}"
                )

        meta_save_path = self.data_root.joinpath(self.dataset_name).joinpath(METADATA_NAME)
        if not meta_save_path.exists():
            file_downloader(META_DATA_URL, meta_save_path.as_posix(), log_error=False)

        self.data_root = self.data_root.joinpath(self.dataset_name)
        meta_data = self.read_meta()

        # prepare npy data if not exists
        if (
            not self.data_root.joinpath(self.ct_folder).exists()
            or not self.data_root.joinpath(self.label_folder).exists()
        ):
            parallel_processing_pipeline(
                self.data_root,
                meta_data,
                self.ct_folder,
                self.label_folder,
                num_readers=4,
                num_processors=1,
                alignment=alignment,
            )

        self.meta_info = self.gen_meta_info()
        self.fold_list = self.gen_fold_list()
        flatten_fold = self.fold_list_flatten(fold_num, fold_index, split)
        self.modality_num = 3
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

    def read_meta(self) -> dict[str, dict[str, list[str]]]:
        raw_meta: list[dict[str, str]]
        meta_data: dict[str, dict[str, list[str]]] = defaultdict(
            lambda: {"ct_path": [], "liver_path": [], "tumor_path": []}
        )
        # meta_data example:
        # {
        #   "patient_id1" : {"ct_path": ["PXXXX_ct_c1.nii.gz", ...], "liver_path": [...], "tumor_path": [...]},
        #   "patient_id2" : {"ct_path": ["PXXXX_ct_c1.nii.gz", ...], "liver_path": [...], "tumor_path": [...]},
        # }
        with open(self.data_root.joinpath(METADATA_NAME), encoding="utf-8", newline="") as csv_file:
            reader = csv.DictReader(csv_file)
            raw_meta = list(reader)

        for info in raw_meta:
            # 排除包含 "_P" 的路徑
            if "_P" in info["ct_path"]:
                continue

            patient_id = info["patient_id"]
            # 如果 patient_id 不存在，defaultdict 會自動執行 lambda 函數創建一個新的字典
            # 使用 [] 鍵存取會自動創建，無需檢查
            meta_data[patient_id]["ct_path"].append(info["ct_path"])
            meta_data[patient_id]["liver_path"].append(info["liver_mask_path"])
            meta_data[patient_id]["tumor_path"].append(info["mask_path"])
        return dict(meta_data)

    def gen_meta_info(self) -> str:
        if self.data_root.joinpath(self.meta_info_name).exists():
            return self.data_root.joinpath(self.meta_info_name).as_posix()
        meta_info = pd.DataFrame(columns=["patient_id", "slice_id", "has_phase0", "has_phase1", "has_phase2"])

        all_ct_patients_ids = sorted(self.data_root.joinpath(self.ct_folder).glob("*"))
        self.logger.info("Generating dataset meta info...")
        for patient_folder in tqdm(all_ct_patients_ids):
            patient_id = patient_folder.name
            phase_0_files = sorted(patient_folder.glob("00_*"))
            phase_1_files = sorted(patient_folder.glob("01_*"))
            phase_2_files = sorted(patient_folder.glob("02_*"))
            for p0_file, p1_file, p2_file in zip(phase_0_files, phase_1_files, phase_2_files, strict=True):
                p0_slice_id = get_slice_id(p0_file.name)
                p1_slice_id = get_slice_id(p1_file.name)
                p2_slice_id = get_slice_id(p2_file.name)
                has_p0 = len(np.unique(np.load(p0_file))) > 1
                has_p1 = len(np.unique(np.load(p1_file))) > 1
                has_p2 = len(np.unique(np.load(p2_file))) > 1
                if not p0_slice_id == p1_slice_id == p2_slice_id:
                    raise IndexError(f"Slice id shold be same, got: {[p0_slice_id, p1_slice_id, p2_slice_id]}")
                meta_info.loc[len(meta_info)] = [patient_id, p0_slice_id, has_p0, has_p1, has_p2]
        meta_info.to_csv(self.data_root.joinpath(self.meta_info_name))
        return self.data_root.joinpath(self.meta_info_name).as_posix()

    def gen_fold_list(self) -> list[dict[str, list[str]]]:
        if self.data_root.joinpath(self.fold_list_name).exists():
            with open(self.data_root.joinpath(self.fold_list_name), encoding="utf-8-sig") as j:
                fold_list = json.load(j)
            return fold_list

        all_ct_patient_ids = sorted(self.data_root.joinpath(self.ct_folder).glob("*"))
        all_label_patient_ids = sorted(self.data_root.joinpath(self.label_folder).glob("*"))

        # last 2 parts of all_image_ids is patient_id/ and slice_id.npy
        mismatch_ids = [
            ("/".join(image_id.parts[-2:]), "/".join(label_id.parts[-2:]))
            for image_id, label_id in zip(all_ct_patient_ids, all_label_patient_ids, strict=True)
            if image_id.name != label_id.name
        ]
        if len(mismatch_ids) > 0:
            raise ValueError(f"The data id of image and label should be same, got: {mismatch_ids}")

        fold_list: list[dict[str, list[str]]] = []
        for fold in chunk_list(all_ct_patient_ids):
            single_fold_dict = {}
            for folder in fold:
                single_fold_dict[folder.name] = sorted(file.name for file in folder.glob("*.npy"))
            fold_list.append(single_fold_dict)
        with open(self.data_root.joinpath(self.fold_list_name), "w", encoding="utf-8-sig") as j:
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
        npy_images_tr_path = self.data_root.joinpath(self.ct_folder)
        npy_labels_tr_path = self.data_root.joinpath(self.label_folder)
        file_pairs = zip(
            sorted(npy_images_tr_path.joinpath(file_path) for file_path in flatten_fold if "00_" in file_path),
            sorted(npy_labels_tr_path.joinpath(file_path) for file_path in flatten_fold if "00_" in file_path),
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
        image_tensor = image_tensor / 255
        label_tensor = tv_tensors.Mask(self.load_image(file_entry["label"]).long())
        if self.transforms is not None:
            image_tensor, label_tensor = self.transforms(image_tensor, label_tensor)

        return image_tensor, label_tensor


def multi_modal_generator(image_path: Path, label_path: Path, modality_num: int) -> dict[str, list[str] | str]:
    case_id = image_path.stem.split("_")[0]
    return {
        "image": [image_path.as_posix().replace("00_", f"{modality_idx:02d}_") for modality_idx in range(modality_num)],
        "label": label_path.as_posix(),
        "case_id": case_id,
    }


def get_slice_id(file_name: str) -> str:
    pattern = r"_(\d{3})\."
    return re.findall(pattern, file_name)[0]
