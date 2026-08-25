from logging import getLogger
from multiprocessing import Manager, Pool
from multiprocessing.pool import AsyncResult
from pathlib import Path
from queue import Queue
from typing import Any

import numpy as np
from tqdm import tqdm

from dynodino.datasets.data_reader import nii_to_npy, nrrd_to_npy

from .alignment_utils import CTData, save_result_to_npy, wawtace_slice_matching, simple_slice_matching


# --- 1. 生產者 (Reader) 任務 ---
def producer(
    patient_id: str,
    data_root: Path,
    ct_files: list[str],
    liver_files: list[str],
    tumor_files: list[str],
) -> CTData:
    try:
        ct_arrays = nii_to_npy(sorted(ct_files), strict=False)
        liver_arrays = nii_to_npy(sorted(liver_files), strict=False)
        tumor_arrays = nrrd_to_npy(sorted(tumor_files), strict=False)
        results: dict[str, Any] = {"patient_id": patient_id, "data_root": data_root}

        for phase_id, (ct_array, liver_array) in enumerate(zip(ct_arrays, liver_arrays, strict=True)):
            if liver_array is not None:
                liver_array = np.rot90(np.where(liver_array == 5, 1, 0).astype(int), 2, axes=(0, 1))
            if ct_array is not None:
                ct_array = np.rot90(ct_array, 2, axes=(0, 1))
            results[f"ct_phase_{phase_id}"] = ct_array
            results[f"label_phase_{phase_id}"] = liver_array

        for phase_id, tumor_array in enumerate(tumor_arrays):
            if results.get(f"ct_phase_{phase_id}") is None:
                if results.get("ct_phase_0") is not None:
                    results[f"ct_phase_{phase_id}"] = np.ones_like(results["ct_phase_0"]) * -1024
            if results.get(f"label_phase_{phase_id}") is None:
                if results.get("label_phase_0") is not None:
                    results[f"label_phase_{phase_id}"] = np.zeros_like(results["label_phase_0"])

            if tumor_array is not None and results.get(f"label_phase_{phase_id}") is not None:
                # The tumor array is loaded with shape (D, H, W).
                # 1. Rotate it 180 degrees on the image plane (Height, Width), which is axes (1, 2).
                tumor_array = np.rot90(tumor_array, 2, axes=(1, 2))

                # 2. Transpose from (D, H, W) to (W, H, D) to match the CT/liver array's shape.
                tumor_array = tumor_array.transpose((2, 1, 0))

                # 3. Overlay the tumor mask onto the label.
                if results[f"label_phase_{phase_id}"].shape == tumor_array.shape:
                    results[f"label_phase_{phase_id}"][tumor_array == 1] = 2
    except Exception as e:
        logger = getLogger("CECT.ct_reader")
        logger.error("Error when reading data of patient: %s, Error: %s", patient_id, e, exc_info=True)

    return CTData(clip_range=(-1024, 1024), **results)


# --- 2. 消費者 (Processor) 任務 ---
def consumer(ct_data: CTData, ct_npy_folder: str, label_npy_folder: str, alignment: bool = True) -> str | Exception:
    """
    消費者任務：接收生產者提供的資料，並進行深度學習預處理。
    """
    try:
        if alignment:
            matched_ct_arrays, matched_label_arrays = wawtace_slice_matching(ct_data)
        else:
            matched_ct_arrays, matched_label_arrays = simple_slice_matching(ct_data)

        ct_data.all_ct = matched_ct_arrays
        ct_data.all_label = matched_label_arrays
        save_result_to_npy(ct_data, ct_npy_folder, label_npy_folder)

        return "success"
    except Exception as e:
        logger = getLogger("CECT.ct_processor")
        logger.error("Error when process patient: %s, Error: %s", ct_data.patient_id, e, exc_info=True)
        return e


def _producer_task(
    queue: Queue,
    patient_id: str,
    data_root: Path,
    ct_files: list[str],
    liver_files: list[str],
    tumor_files: list[str],
):
    try:
        ct_data = producer(patient_id, data_root, ct_files, liver_files, tumor_files)
        queue.put(ct_data)
    except Exception as e:
        queue.put(e)


def _consumer_task(
    queue: Queue, progress_queue: Queue, ct_npy_folder: str, label_npy_folder: str, alignment: bool = True
):
    results: list[str | Exception] = []
    while True:
        data = queue.get()
        if data is None:  # Sentinel
            break

        if isinstance(data, Exception):
            results.append(data)
            progress_queue.put(1)
            continue

        result = consumer(data, ct_npy_folder, label_npy_folder, alignment=alignment)
        results.append(result)
        progress_queue.put(1)
    return results


# --- 3. 主要管線控制 ---
def single_process_pipeline(
    data_root: Path,
    metadata: dict[str, dict[str, list[str]]],
    ct_npy_folder: str,
    label_npy_folder: str,
    alignment: bool = True,
):
    for patient_id, meta in metadata.items():
        ct_files = [data_root.joinpath(path).as_posix() for path in sorted(meta["ct_path"])]
        liver_files = [data_root.joinpath(path).as_posix() for path in sorted(meta["liver_path"])]
        tumor_files = [data_root.joinpath(path).as_posix() for path in sorted(meta["tumor_path"])]
        ct_data = producer(patient_id, data_root, ct_files, liver_files, tumor_files)
        consumer(ct_data, ct_npy_folder, label_npy_folder, alignment=alignment)


def parallel_processing_pipeline(
    data_root: Path,
    metadata: dict[str, dict[str, list[str]]],
    ct_npy_folder: str,
    label_npy_folder: str,
    num_readers: int,
    num_processors: int,
    alignment: bool = True,
):
    """
    主要執行函數，設置兩個 Pool：生產者 (Reader) 和 消費者 (Processor)。
    """
    logger = getLogger("CECT.processing_pipeline")
    logger.info("--- Start to process CECT CT Data ---")
    logger.info(" Alignment mode: %s", "Enabled" if alignment else "Disabled")
    logger.info(" Number of data reader process: %d", num_readers)
    logger.info(" Number of data processor process: %d", num_processors)

    manager = Manager()
    data_queue = manager.Queue(maxsize=8)
    progress_queue = manager.Queue()
    total_tasks = len(metadata)

    def _monitor_consumer_progress(pbar: tqdm, total: int):
        for _ in range(total):
            pbar.update(progress_queue.get())

    with Pool(processes=num_readers) as reader_pool, Pool(processes=num_processors) as processor_pool:
        with (
            tqdm(total=total_tasks, desc="Data Reading   ", position=0) as p_pbar,
            tqdm(total=total_tasks, desc="Data Processing", position=1) as c_pbar,
        ):
            monitor = __import__("threading").Thread(target=_monitor_consumer_progress, args=(c_pbar, total_tasks))
            monitor.start()
            # --- 提交消費者任務 ---
            processor_results = [
                processor_pool.apply_async(
                    _consumer_task,
                    (data_queue, progress_queue, ct_npy_folder, label_npy_folder, alignment),
                )
                for _ in range(num_processors)
            ]

            # --- 提交生產者任務 ---
            reader_futures: list[AsyncResult] = []
            for patient_id, meta in metadata.items():
                ct_files = [data_root.joinpath(path).as_posix() for path in sorted(meta["ct_path"])]
                liver_files = [data_root.joinpath(path).as_posix() for path in sorted(meta["liver_path"])]
                tumor_files = [data_root.joinpath(path).as_posix() for path in sorted(meta["tumor_path"])]
                future = reader_pool.apply_async(
                    _producer_task,
                    (data_queue, patient_id, data_root, ct_files, liver_files, tumor_files),
                    callback=lambda _: p_pbar.update(1),
                )
                reader_futures.append(future)

            # 等待所有生產者任務完成
            for future in reader_futures:
                future.get()

            # 當所有生產者任務都完成後，發送停止信號給消費者
            for _ in range(num_processors):
                data_queue.put(None)

            # --- 等待所有消費者任務完成 ---
            logger.info("Waiting for all processing tasks to complete...")
            monitor.join()

            all_results = []
            for res in processor_results:
                all_results.extend(res.get())

            task_num: int = 0
            fail_num: int = 0
            for result_summary in all_results:
                task_num += 1
                if isinstance(result_summary, Exception):
                    fail_num += 1
            if task_num > 0:
                logger.info(
                    "%d task done with %d fail, fail rate: %.2f%%.", task_num, fail_num, (fail_num / task_num) * 100
                )
            else:
                logger.info("No task was processed.")
