import logging
import random
import traceback
from collections.abc import Generator
from logging import Logger, getLogger
from typing import TypeVar

import numpy as np
import requests
import torch
from tqdm.auto import tqdm

T = TypeVar("T")


def log_traceback(exc: Exception, logger: Logger | None):
    """Log the traceback of an exception using the provided logger."""
    tb_str = "".join(traceback.format_exception(exc))
    logger = logger or getLogger("dynodino.utils")
    orig_formatters = [handler.formatter for handler in logger.handlers]
    exc_formatter = logging.Formatter("\t >> %(message)s")
    for handler in logger.handlers:
        handler.setFormatter(exc_formatter)
    for line in tb_str.strip().split("\n"):
        if "The above exception" in line:
            logger.error("")
            logger.error(line)
            logger.error("")
        else:
            logger.error(line)
    for handler, orig_formatter in zip(logger.handlers, orig_formatters, strict=False):
        handler.setFormatter(orig_formatter)


def file_downloader(file_url: str, save_path: str, log_error: bool = False):
    """Download the model from the specified URL and save it to the given path."""
    logger = getLogger("dynodino.file_downloader")
    logger.info(f"Downloading file from {file_url} to {save_path}")
    response = requests.get(file_url, stream=True, timeout=10)
    try:
        response.raise_for_status()
    except requests.HTTPError as e:
        logger.error(f"Failed to download file: {e}")
        if log_error:
            log_traceback(e, logger)
        raise e

    total_size = int(response.headers.get("content-length", 0))
    block_size = 1024  # 1 KB
    size_unit = "KB"
    size_divisor = 1024
    match total_size:
        case size if size >= 1024**3:
            size_divisor = 1024**3
            size_unit = "GB"
        case size if size >= 1024**2:
            size_divisor = 1024**2
            size_unit = "MB"
    logger.info(f"Total file size: {total_size / size_divisor:.2f} {size_unit}")

    with tqdm(desc=save_path, total=total_size, unit="iB", unit_scale=True, unit_divisor=1024) as pbar:
        with open(save_path, "wb") as file:
            for chunk in response.iter_content(block_size):
                if chunk:  # 過濾掉 keep-alive 的空資料塊
                    file.write(chunk)
                    pbar.update(len(chunk))  # 更新進度條
    logger.info("Download completed.")


def chunk_list(input_list: list[T], num_chunks: int = 10) -> Generator[list[T]]:
    """
    Splits the input list into num_chunks chunks of approximately equal size and yields them as a generator.

    - The total number of chunks yielded is fixed at 10.
    - The concatenation of all yielded chunks contains all elements of the input_list.
    - Chunk sizes are as close to equal as possible (difference of at most 1 element).
    - If the length of input_list is > 10, all yielded chunks are guaranteed to be non-empty.

    Args:
        input_list: A list of any length.
        num_chunks: number to chunk the input list.

    Yields:
        List[Any]: A portion (chunk) of the input_list.
    """

    list_len = len(input_list)
    base_size = list_len // num_chunks
    remainder = list_len % num_chunks
    end_index = 0
    current_index = 0

    for i in range(num_chunks):
        match base_size + (1 if i < remainder else 0):
            case 0:
                yield []
            case chunk_size:
                # get the current chunk
                end_index = current_index + chunk_size
                chunk = input_list[current_index:end_index]
                yield chunk
                current_index = end_index


def set_seed(seed: int) -> None:
    """Sets the seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
