import json
import re
import shutil
from collections import defaultdict
from logging import getLogger
from pathlib import Path
from typing import Literal, cast

import nrrd
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import tv_tensors
from torchvision.transforms import v2
from tqdm.auto import tqdm

from data_preprocessing.data_reader import nrrd_to_npy
from data_preprocessing.preprocess_lib.alignment_lib.wawtace_alignment import (
    parallel_processing_pipeline,
)
from utils.file_utils import extract_data_file, folder_validate
from utils.utils import chunk_list, file_downloader

DATA_URL = (
    "https://zenodo.org/records/12741586/files/ct_scans_1_4_wawtace_09_05_24.zip?download=1",
    "https://zenodo.org/records/12741586/files/ct_scans_2_4_wawtace_09_05_24.zip?download=1",
    "https://zenodo.org/records/12741586/files/ct_scans_3_4_wawtace_09_05_24.zip?download=1",
    "https://zenodo.org/records/12741586/files/ct_scans_4_4_wawtace_09_05_24.zip?download=1",
    "https://zenodo.org/records/12741586/files/organ_masks_wawtace_09_05_2024.zip?download=1",
    "https://zenodo.org/records/12741586/files/tumor_masks_wawtace_v1_08_05_2024.zip?download=1",
)
TUMOR_MASK_FOLDER = "tumor_masks_wawtace_v1_08_05_2024"
CT_NII_FOLDER = "ct_nii_files"
CT_NPY_FOLDER = "new_ct_npy_files"
LABEL_NII_FOLDER = "label_nii_files"
LABEL_NPY_FOLDER = "new_label_npy_files"
META_INFO_NAME = "meta_info.csv"
FOLD_LIST_NAME = "fold_list.json"


class WAWTACEDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(
        self,
        data_root: str,
        num_workers: int = 4,
        split: Literal["train", "test"] = "train",
        fold_num: Literal[5, 10] = 5,
        fold_index: int = 0,
        ct_window: tuple[int, int] = (-1024, 1024),
        transform: v2.Transform | None = None,
        alignment: bool = True,
    ):
        """WAWTACEDataset will automatically download the data from *WAW-TACE* official source.

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
        self.logger = getLogger("WAWTACEDataset")
        self.dataset_name = "WAWTACE"
        self.data_root = Path(data_root)
        self.ct_window = ct_window
        self.alignment = alignment
        self.ct_folder = CT_NPY_FOLDER if alignment else "unaligned_ct_npy_files"
        self.label_folder = LABEL_NPY_FOLDER if alignment else "unaligned_label_npy_files"
        self.meta_info_name = META_INFO_NAME if alignment else "unaligned_meta_info.csv"
        self.fold_list_name = FOLD_LIST_NAME if alignment else "unaligned_fold_list.json"
        folder_validate(self.data_root)
        if fold_index > fold_num - 1:
            raise IndexError("fold_index out of range, fold_index should be less than fold_num - 1.")

        if not self.data_root.joinpath(self.dataset_name).exists():
            # If the data folder already exist, do nothing
            self.data_root.joinpath(self.dataset_name).mkdir(exist_ok=True)

        loaded_files: list[Path] = []
        for url in DATA_URL:
            # download and extract the dataset
            file_name = url.split("/")[-1].split("?")[0]
            zip_save_path = self.data_root.joinpath(self.dataset_name).joinpath(file_name)
            if not zip_save_path.exists():
                file_downloader(url, zip_save_path.as_posix(), log_error=False)
            else:
                self.logger.info("File: %s already exist, skip download.", zip_save_path.name)
            loaded_files.append(zip_save_path)

        self.data_root = self.data_root.joinpath(self.dataset_name)

        self.extract_data(CT_NII_FOLDER, loaded_files, True)
        self.extract_data(LABEL_NII_FOLDER, loaded_files, False)

        # Merge labels of different tumor in same phase
        if self.data_root.joinpath(LABEL_NII_FOLDER).joinpath(TUMOR_MASK_FOLDER).exists():
            for tumor_folder in self.data_root.joinpath(LABEL_NII_FOLDER).joinpath(TUMOR_MASK_FOLDER).glob("*"):
                folder_name = tumor_folder.stem
                for tumor_file in tumor_folder.glob("*"):
                    shutil.move(tumor_file, self.data_root.joinpath(LABEL_NII_FOLDER).joinpath(folder_name))
                tumor_folder.rmdir()
            self.data_root.joinpath(LABEL_NII_FOLDER).joinpath(TUMOR_MASK_FOLDER).rmdir()

        self.clean_label_filenames()

        # prepare npy data if not exists
        self.logger.info("Merging tumor nrrd label file...")
        self.merge_nrrd()
        self.logger.info("Merge done.")
        meta_data = self.read_meta()
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

    def extract_data(self, target_folder: str, loaded_files: list[Path], is_ct_scans: bool):
        if not self.data_root.joinpath(target_folder).exists():
            self.logger.info("Start to extract files.")
            self.data_root.joinpath(target_folder).mkdir(exist_ok=True)
            extracted_files: list[str] = []
            for zip_save_path in loaded_files:
                if ("ct_scans" in zip_save_path.name) is is_ct_scans:
                    self.logger.info("Extracting file: %s", zip_save_path.name)
                    extracted_files.extend(extract_data_file(zip_save_path.as_posix(), mkdir=False))
            for file in extracted_files:
                shutil.move(file, self.data_root.joinpath(target_folder))

    def clean_label_filenames(self):
        """
        清洗資料集檔名。
        1. 忽略 .nrrd。
        2. 將 .nii.gz 統一格式為 <id>_<index>_total_segmentator.nii.gz
        """

        # 使用 rglob 遞迴搜尋所有檔案
        # 因為 .nii.gz 會被 glob 判定為以 .gz 結尾，這裡直接抓所有檔案後過濾
        for file_path in self.data_root.joinpath(LABEL_NII_FOLDER).rglob("*"):
            # 1. 忽略資料夾與 .nrrd 檔案
            if file_path.is_dir() or file_path.suffix == ".nrrd":
                continue

            # 2. 處理 .nii.gz 檔案 (包含 .nii.nii.gz 的情況)
            if "".join(file_path.suffixes).endswith(".nii.gz"):
                parts = file_path.name.split("_")

                # 根據結構，前四個部分應該是 id, index, total, segmentator
                # 例如: ['358', '0', 'total', 'segmentator.1', ...]
                patient_id = parts[0]
                phase = parts[1]
                new_name = f"{patient_id}_{phase}_total_segmentator.nii.gz"
                if file_path.name != new_name:
                    # 建立新的檔名
                    new_path = file_path.with_name(new_name)

                    # 執行重命名 (rename)
                    try:
                        file_path.rename(new_path)
                        self.logger.info("Renamed: %s -> %s success.", file_path.name, new_name)
                    except Exception as e:
                        self.logger.error("Error renaming %s: %s", file_path.name, e)

    def merge_array(self, arrays: list[np.ndarray] | list[None]) -> np.ndarray | None:
        """比較 list 中所有二維陣列的 shape，若相同則以 'OR' 邏輯合併。

        Args:
            arrays (list[np.ndarray]): 包含多個 0, 1 二維陣列的列表。

        Returns:
            np.ndarray: 合併後的單一二維陣列。

        """
        # 1. 基本檢查：如果 list 為空
        if len(arrays) == 0:
            return None
        # 2. 若只有一個，直接回傳
        if len(arrays) == 1:
            return arrays[0]
        if any(ele is None for ele in arrays):
            return None
        valid_arrays = cast(list[np.ndarray], arrays)

        # 3. 檢查所有陣列的 shape 是否一致
        first_shape = valid_arrays[0].shape
        for i, arr in enumerate(valid_arrays[1:], start=1):
            if arr.shape != first_shape:
                self.logger.warning(f"陣列 Shape 不一致！索引 0 為 {first_shape}，但索引 {i} 為 {arr.shape}")
                return None
        # 4. 執行 OR 邏輯合併
        # np.bitwise_or.reduce 會對 list 中的所有陣列進行元素級別的 OR 運算
        merged_array = np.bitwise_or.reduce(valid_arrays)

        return merged_array

    def merge_nrrd(self) -> None:
        label_nii_path = self.data_root.joinpath(LABEL_NII_FOLDER)
        for patient_folder in tqdm(sorted(label_nii_path.glob("*"))):
            if len(list(patient_folder.glob("*_merged_tumor_seg.nrrd"))) > 0:
                continue
            patient_id = patient_folder.name
            for phase_id in range(1, 4):
                nrrd_arrays = nrrd_to_npy(
                    [
                        nrrd_file.as_posix()
                        for nrrd_file in patient_folder.glob("*.nrrd")
                        if nrrd_file.name.startswith(f"{patient_id}_{phase_id}")
                    ],
                    strict=True,
                )

                merged_nrrd = self.merge_array(nrrd_arrays)
                if merged_nrrd is not None:
                    nrrd.write(
                        patient_folder.joinpath(f"{patient_id}_{phase_id}_merged_tumor_seg.nrrd").as_posix(),
                        merged_nrrd,
                    )

    def read_meta(self) -> dict[str, dict[str, list[str]]]:
        ct_name_template = "{patient_id}_{phase_id}_scan.nii.gz"
        liver_name_template = "{patient_id}_{phase_id}_total_segmentator.nii.gz"
        tumor_name_template = "{patient_id}_{phase_id}_merged_tumor_seg.nrrd"
        meta_data: dict[str, dict[str, list[str]]] = defaultdict(
            lambda: {"ct_path": [], "liver_path": [], "tumor_path": []}
        )
        # meta_data example:
        # {
        #   "patient_id1" : {"ct_path": ["PXXXX_ct_c1.nii.gz", ...], "liver_path": [...], "tumor_path": [...]},
        #   "patient_id2" : {"ct_path": ["PXXXX_ct_c1.nii.gz", ...], "liver_path": [...], "tumor_path": [...]},
        # }
        ct_nii_path = self.data_root.joinpath(CT_NII_FOLDER)
        label_nii_path = self.data_root.joinpath(LABEL_NII_FOLDER)
        for patient_ct_folder in sorted(ct_nii_path.glob("*")):
            if not label_nii_path.joinpath(patient_ct_folder.name).exists():
                self.logger.warning("The patient: %s does not have label file.", patient_ct_folder.name)
                continue
            patient_label_folder = label_nii_path.joinpath(patient_ct_folder.name)
            patient_id = patient_ct_folder.name
            meta_data[patient_ct_folder.name]["ct_path"] = [
                Path(*p.parts[-3:]).as_posix()
                for p in patient_ct_folder.glob("*")
                if not p.name.startswith(f"{patient_id}_0_")
            ]
            meta_data[patient_ct_folder.name]["liver_path"] = [
                Path(*p.parts[-3:]).as_posix()
                for p in patient_label_folder.glob("*")
                if "nii.gz" in p.name and not p.name.startswith(f"{patient_id}_0_")
            ]
            meta_data[patient_ct_folder.name]["tumor_path"] = [
                Path(*p.parts[-3:]).as_posix()
                for p in patient_label_folder.glob("*.nrrd")
                if "merged" in p.name and not p.name.startswith(f"{patient_id}_0_")
            ]

        pop_patient_ids = []
        for patient_id, meta in meta_data.items():
            for phase_id in range(1, 4):
                tumor_name = tumor_name_template.format(patient_id=patient_id, phase_id=phase_id)
                tumor_path = Path(*label_nii_path.joinpath(patient_id).joinpath(tumor_name).parts[-3:]).as_posix()
                ct_name = ct_name_template.format(patient_id=patient_id, phase_id=phase_id)
                ct_path = Path(*ct_nii_path.joinpath(patient_id).joinpath(ct_name).parts[-3:]).as_posix()
                liver_name = liver_name_template.format(patient_id=patient_id, phase_id=phase_id)
                liver_path = Path(*label_nii_path.joinpath(patient_id).joinpath(liver_name).parts[-3:]).as_posix()
                if phase_id == 1 and (
                    tumor_path not in meta["tumor_path"]
                    or ct_path not in meta["ct_path"]
                    or liver_path not in meta["liver_path"]
                ):
                    pop_patient_ids.append(patient_id)
                    continue
                if tumor_path not in meta["tumor_path"]:
                    meta["tumor_path"].append(tumor_path.replace(".nrrd", ".NA"))
                if ct_path not in meta["ct_path"]:
                    meta["ct_path"].append(ct_path.replace(".nii.gz", ".NA"))
                if liver_path not in meta["liver_path"]:
                    meta["liver_path"].append(liver_path.replace(".nii.gz", ".NA"))

        for pop_id in pop_patient_ids:
            meta_data.pop(pop_id)

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
            image_tensor = torch.from_numpy(np.load(image_path)).unsqueeze(0).float()
            if image_tensor.shape != (1, 512, 512):
                image_tensor = (
                    torch.nn.functional.interpolate(image_tensor.unsqueeze(0), (512, 512), mode="nearest")
                    .squeeze(0)
                    .long()
                )
        else:
            image_slices = [torch.from_numpy(np.load(modality_path)) for modality_path in image_path]

            for idx in range(len(image_slices)):
                if image_slices[idx].shape != (512, 512):
                    img = image_slices[idx].unsqueeze(0).unsqueeze(0).float()
                    image_slices[idx] = (
                        torch.nn.functional.interpolate(img, (512, 512), mode="nearest").squeeze().short()
                    )
            image_tensor = torch.stack(image_slices, dim=0)

        return image_tensor

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        image_tensor: torch.Tensor
        label_tensor: torch.Tensor
        file_entry = self.file_list[idx]
        # print(file_entry)
        image_tensor = self.load_image(file_entry["image"]).float()
        image_tensor = (image_tensor.clip(*sorted(self.ct_window)) - min(self.ct_window)) / (
            max(self.ct_window) - min(self.ct_window)
        )
        label_tensor = tv_tensors.Mask(self.load_image(file_entry["label"]).long())
        if self.transforms is not None:
            image_tensor, label_tensor = self.transforms(image_tensor, label_tensor)

        return image_tensor, label_tensor


def multi_modal_generator(image_path: Path, label_path: Path, modality_num: int) -> dict[str, list[str] | str]:
    return {
        "image": [image_path.as_posix().replace("00_", f"{modality_idx:02d}_") for modality_idx in range(modality_num)],
        "label": label_path.as_posix(),
    }


def get_slice_id(file_name: str) -> str:
    pattern = r"_(\d{3})\."
    return re.findall(pattern, file_name)[0]
