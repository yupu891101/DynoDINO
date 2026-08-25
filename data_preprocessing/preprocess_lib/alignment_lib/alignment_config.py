import os
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class DataPath:
    """Setting the path to the converted .npy files.
    the folder structure should be:
    <ct_npy_file_path>/
        ├── <PatientId>/
        │   ├── <PahseId:2d>_<SliceId:3d>.npy
        │   ├── 00_001.npy
        │   ├── 01_001.npy
        │   ...
        ├── <PatientId>/
        │   ├── <PahseId:2d>_<SliceId:3d>.npy
        │   ├── 00_001.npy
        │   ├── 01_001.npy
        │   ...
        ...
    <label_npy_file_path>/
        ├── <PatientId>/
        │   ├── <PahseId:2d>_<SliceId:3d>.npy
        │   ├── 00_001.npy
        │   ├── 01_001.npy
        │   ...
        ├── <PatientId>/
        │   ├── <PahseId:2d>_<SliceId:3d>.npy
        │   ├── 00_001.npy
        │   ├── 01_001.npy
        │   ...
        ...
    """

    ct_npy_file_path: Path = Path("../Data_Disk/CECT/ct_npy_files/")
    label_npy_file_path: Path = Path("../Data_Disk/CECT/label_npy_files/")


@dataclass
class EqualizeConfig:
    _value_range: tuple[int, int]
    kernel_size_ratio: float = 1 / 16
    clip_limit: float = 0.005
    nbins: int = field(init=False, default=0)

    def __post_init__(
        self,
    ):
        self.nbins = max(self._value_range) - min(self._value_range)


@dataclass
class SliceAlignmentConfig:
    ct_npy_file_path: Path = DataPath.ct_npy_file_path
    label_npy_file_path: Path = DataPath.label_npy_file_path

    num_producer: int = 4
    num_producer_thread: int = 4
    num_consumer: int = 16
    max_dicom_queue: int = 8

    reference_percentile: int | float = 75
    align_slice_percentile: tuple[int | float, ...] = tuple({5, 25, 50, 75, 95})
    padding_value: int = 0
    split_num: int = 5
    top_k_offset: int = 2
