from logging import getLogger
from multiprocessing import Pool
from multiprocessing.pool import AsyncResult
from pathlib import Path
from typing import overload

import nibabel as nib
import nrrd
import numpy as np
import SimpleITK as Sitk
from tqdm.auto import tqdm

from dynodino.utils.file_utils import folder_validate

logger = getLogger(__name__)


def read_nii_sitk(nii_file_path: str) -> np.ndarray:
    """
    使用 SimpleITK 讀取 .nii 或 .nii.gz 檔案並轉換為 NumPy 陣列。

    參數:
        nii_file_path (str): .nii 或 .nii.gz 檔案的路徑。

    回傳:
        np.ndarray: 讀取的影像資料轉換成的 NumPy 陣列。
    """
    # 讀取 NIfTI 檔案
    itk_image = Sitk.ReadImage(nii_file_path)
    # 將 ITK 影像轉換為 NumPy 陣列
    numpy_array = Sitk.GetArrayFromImage(itk_image)
    return numpy_array


def read_nii_nib(nii_file_path: str) -> np.ndarray:
    """
    使用 NiBabel 讀取 .nii 或 .nii.gz 檔案並轉換為 NumPy 陣列。

    參數:
        nii_file_path (str): .nii 或 .nii.gz 檔案的路徑。

    回傳:
        np.ndarray: 讀取的影像資料轉換成的 NumPy 陣列。
    """
    # 讀取 NIfTI 檔案
    nii_image = nib.loadsave.load(nii_file_path)
    # 將影像資料轉換為 NumPy 陣列
    numpy_array = nii_image.get_fdata()
    return numpy_array


READ_NII_BACKEND = {"sitk": read_nii_sitk, "nib": read_nii_nib}


def msd_data_saver(nii_file_path: str, npy_file_path: str, backend: str = "sitk") -> None | Exception:
    """
    將 .nii 或 .nii.gz 檔案轉換為 NumPy 陣列。

    參數:
        nii_file_path (str): .nii 或 .nii.gz 檔案的路徑。
        engine (str): 使用的讀取引擎，"sitk" 或 "nib"。預設為 "sitk"。

    回傳:
        np.ndarray: 讀取的影像資料轉換成的 NumPy 陣列。
    """
    Path(npy_file_path).mkdir(parents=True, exist_ok=True)
    if backend not in READ_NII_BACKEND:
        return ValueError("Unsupported engine. Use 'sitk' or 'nib'.")
    try:
        npy_data = READ_NII_BACKEND[backend](nii_file_path)
        if len(npy_data.shape) == 3:
            for i in range(npy_data.shape[0]):
                npy_slice = npy_data[i, :, :]
                save_path = f"{npy_file_path}/{i:03d}.npy"
                np.save(save_path, npy_slice)
        elif len(npy_data.shape) == 4:
            for p in range(npy_data.shape[0]):
                for i in range(npy_data.shape[1]):
                    npy_slice = npy_data[p, i, :, :]
                    save_path = f"{npy_file_path}/{p:02d}_{i:03d}.npy"
                    np.save(save_path, npy_slice)
        else:
            raise ValueError(f"Unsupported NIfTI data shape: {npy_data.shape}, expected 3D or 4D.")
    except Exception as e:
        return e


def msd_data_converter(images_root: Path, target_folder: Path, num_workers: int = 4):
    folder_validate(images_root)
    logger.info("Converting NIfTI files in %s to NumPy files into %s", images_root, target_folder)
    target_folder.mkdir(parents=True, exist_ok=True)
    worker_args: list[tuple[str, str]] = []
    for file in images_root.glob("[!.]*.nii.gz"):
        worker_args.append((file.as_posix(), target_folder.joinpath(file.stem.split(".")[0]).as_posix()))

    pbar = tqdm(total=len(worker_args), desc="Converting NIfTI to NumPy", unit="file")

    def update_pbar(result):
        """Callback function to update tqdm after a task successfully completes."""
        pbar.update(1)

    # 使用多進程進行轉換
    futures: list[AsyncResult[Exception | None]] = []
    with Pool(processes=num_workers) as pool:
        for args in worker_args:
            future = pool.apply_async(msd_data_saver, args=args, callback=update_pbar)
            futures.append(future)

        # 檢查是否有任何轉換過程中出現的異常
        pool.close()
        pool.join()
        for future in futures:
            result = future.get()
            if isinstance(result, Exception):
                logger.error("Error occurred while converting NIfTI to NumPy: %s", result)
                raise result
    pbar.close()


@overload
def nii_to_npy(nii_files: list[str], backend: str = "sitk", strict: bool = True) -> list[np.ndarray] | list[None]: ...
@overload
def nii_to_npy(nii_files: list[str], backend: str = "sitk", strict: bool = False) -> list[np.ndarray | None]: ...


def nii_to_npy(nii_files: list[str], backend: str = "sitk", strict: bool = True):
    if backend not in READ_NII_BACKEND:
        raise ValueError("Unsupported engine. Use 'sitk' or 'nib'.")
    read_func = READ_NII_BACKEND[backend]
    if strict:
        if any(file.endswith("NA") for file in nii_files):
            return [None] * len(nii_files)
        else:
            return [read_func(file) for file in nii_files]
    else:
        return_arrays: list[np.ndarray | None] = []
        for file in nii_files:
            if file.endswith("NA"):
                return_arrays.append(None)
            else:
                return_arrays.append(read_func(file))
        return return_arrays


@overload
def nrrd_to_npy(nrrd_files: list[str], strict: bool = True) -> list[np.ndarray] | list[None]: ...
@overload
def nrrd_to_npy(nrrd_files: list[str], strict: bool = False) -> list[np.ndarray | None]: ...


def nrrd_to_npy(nrrd_files: list[str], strict: bool = True):
    if strict:
        if any(file.endswith("NA") for file in nrrd_files):
            return [None] * len(nrrd_files)
        else:
            strict_arrays: list[np.ndarray] = []
            for nrrd_file in nrrd_files:
                nrrd_image, _ = nrrd.read(nrrd_file)
                strict_arrays.append(nrrd_image)
        return strict_arrays
    else:
        return_arrays: list[np.ndarray | None] = []
        for nrrd_file in nrrd_files:
            if nrrd_file.endswith("NA"):
                return_arrays.append(None)
            else:
                nrrd_image, _ = nrrd.read(nrrd_file)
                return_arrays.append(nrrd_image)
        return return_arrays


def msd_base_data_reader(
    task_root: str,
    image_npy_folder_name: str,
    label_npy_folder_name: str,
    num_workers: int = 4,
):
    task_root_path = Path(task_root)
    folder_validate(task_root_path)
    train_images = task_root_path.joinpath("imagesTr")
    npy_train_images = task_root_path.joinpath(image_npy_folder_name)
    train_labels = task_root_path.joinpath("labelsTr")
    npy_train_labels = task_root_path.joinpath(label_npy_folder_name)
    logger.info("Starting conversion of training images...")
    msd_data_converter(train_images, npy_train_images, num_workers)
    logger.info("Starting conversion of training labels...")
    msd_data_converter(train_labels, npy_train_labels, num_workers)
    logger.info("Data conversion completed.")
