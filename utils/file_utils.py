import logging
import subprocess
from collections.abc import Callable
from pathlib import Path

logger = logging.getLogger(__name__)


def folder_validate(folder_path: Path) -> bool:
    """
    驗證給定的資料夾路徑是否存在且為目錄。

    參數:
    folder_path (str): 資料夾的完整路徑。

    回傳:
    bool: 如果資料夾存在且為目錄，則回傳 True, 否則回傳 False。
    """
    if not folder_path.is_dir():
        logger.error("Folder path %s is not a valid directory.", folder_path)
        raise ValueError(f"Folder path {folder_path} is not a valid directory.")
    if not folder_path.exists():
        logger.error("Folder path %s does not exist.", folder_path)
        raise FileNotFoundError(f"Folder path {folder_path} does not exist.")
    return True


def extract_data_file(data_file_path: str, mkdir: bool = False) -> list[str]:
    """
    從給定的壓縮檔案路徑中提取資料檔案名稱（不含副檔名）。

    參數:
    date_file_path (str): 壓縮檔案的完整路徑。

    回傳:
    str: 資料檔案名稱（不含副檔名）。
    """
    file_path = Path(data_file_path)
    file_suffix = file_path.suffix.lower()
    file_extractor: dict[str, Callable[[str, bool], list[str] | None]] = {
        ".tar": untar_with_cli_progress,
        ".zip": unzip_with_cli_progress,
    }
    extractor = file_extractor.get(file_suffix)

    if extractor:
        extracted_files = extractor(data_file_path, mkdir)
        if extracted_files:
            return extracted_files
        else:
            logger.warning("No files were extracted from %s", data_file_path)
            return []
    else:
        logger.error("Unsupported file format: %s", file_suffix)
        raise ValueError(f"Unsupported file format: {file_suffix}")


def extract_file_diff(original_files: set[str], extract_dir: Path) -> list[str]:
    """
    比較解壓縮前後目標目錄中的檔案，找出新解壓縮出來的檔案列表。

    參數:
    original_files (set[str]): 解壓縮前目標目錄中的檔案絕對路徑集合。
    extract_dir (Path): 解壓縮目標目錄的 Path 物件。

    回傳:
    list[str]: 新解壓縮出來的檔案絕對路徑列表。
    """
    current_files = {file_object.absolute().as_posix() for file_object in extract_dir.glob("*")}
    new_files = current_files - original_files
    return list(new_files)


def untar_with_cli_progress(tar_file_path: str, mkdir: bool = False):
    """
    使用 Python 的 subprocess 呼叫 Ubuntu 的 'tar' 指令進行解壓縮，
    並即時讀取 tar 的 stdout (檔案列表) 作為進度提示。

    參數:
    tar_file_path (str): 待解壓縮的 .tar 壓縮檔案的完整路徑。
    """
    # 1. 路徑處理和檢查
    file_path = Path(tar_file_path).resolve()
    if not file_path.is_file():
        logger.error(f"File not found or invalid path: {tar_file_path}")
        return

    # 確定解壓縮的目標目錄
    if not mkdir:
        extract_dir = file_path.parent
        extract_dir.mkdir(parents=True, exist_ok=True)  # 確保目標目錄存在
    else:
        extract_dir = file_path.parent.joinpath(file_path.stem.split(".")[0])
        extract_dir.mkdir(parents=True, exist_ok=True)  # 確保目標目錄存在
    original_files = {file_object.absolute().as_posix() for file_object in extract_dir.glob("*")}

    logger.info("starting to extract tar file: %s", tar_file_path)
    logger.info("target directory: %s", extract_dir)
    logger.info("-" * 30)

    command = ["tar", "-xvf", str(file_path), "-C", str(extract_dir)]

    try:
        # 3. 使用 Popen 啟動子進程並實時讀取輸出
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,  # 捕捉標準輸出
            stderr=subprocess.STDOUT,  # 將標準錯誤導向標準輸出，確保所有輸出都被捕捉
            text=True,  # 以文本模式讀取輸出
            bufsize=1,  # 使用行緩衝，以便即時讀取
        )

        # 4. 即時讀取和顯示輸出
        # 讀取 tar -v 輸出的每一個檔案路徑
        if process.stdout is None:
            logger.warning("Failed to capture stdout from tar process.")
        else:
            for line in process.stdout:
                # 去除首尾空白，包括換行符
                clean_line = line.strip()
                if clean_line:
                    # 打印當前正在解壓縮的檔案
                    print(f"\r\033[2K extract --> {clean_line}", end="", flush=True)
            print()

        # 5. 等待子進程完成並獲取返回碼
        process.wait()

        # 6. 檢查返回碼以確定是否成功
        if process.returncode == 0:
            logger.info("-" * 30)
            logger.info("Extraction completed successfully, all files saved to %s.", extract_dir)
            return extract_file_diff(original_files, extract_dir)
        else:
            logger.error("-" * 30)
            # 如果失敗，通常錯誤訊息已在 stdout/stderr 中輸出
            logger.error("tar command failed with return code %d", process.returncode)

    except FileNotFoundError:
        logger.error("tar command not found. Please ensure tar is installed on your system.")
    except Exception as e:
        logger.error("Unknown error occurred while executing subprocess: %s", e)


def unzip_with_cli_progress(zip_file_path: str, mkdir: bool = False):
    """
    使用 Python 的 subprocess 呼叫 Ubuntu 的 'unzip' 指令進行解壓縮，
    並即時讀取 unzip 的 stdout (檔案列表) 作為進度提示。

    參數:
    zip_file_path (str): 待解壓縮的 .zip 壓縮檔案的完整路徑。
    """
    # 1. 路徑處理和檢查
    file_path = Path(zip_file_path).resolve()
    if not file_path.is_file():
        logger.error(f"File not found or invalid path: {zip_file_path}")
        return

    # 確定解壓縮的目標目錄
    if not mkdir:
        extract_dir = file_path.parent
        extract_dir.mkdir(parents=True, exist_ok=True)  # 確保目標目錄存在
    else:
        extract_dir = file_path.parent.joinpath(file_path.stem.split(".")[0])
        extract_dir.mkdir(parents=True, exist_ok=True)  # 確保目標目錄存在
    original_files = {file_object.absolute().as_posix() for file_object in extract_dir.glob("*")}

    logger.info("starting to extract zip file: %s", zip_file_path)
    logger.info("target directory: %s", extract_dir)
    logger.info("-" * 30)

    command = ["unzip", "-o", str(file_path), "-d", str(extract_dir)]

    try:
        # 3. 使用 Popen 啟動子進程並實時讀取輸出
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,  # 捕捉標準輸出
            stderr=subprocess.STDOUT,  # 將標準錯誤導向標準輸出，確保所有輸出都被捕捉
            text=True,  # 以文本模式讀取輸出
            bufsize=1,  # 使用行緩衝，以便即時讀取
        )

        # 4. 即時讀取和顯示輸出
        # 讀取 unzip 輸出的每一個檔案路徑
        if process.stdout is None:
            logger.warning("Failed to capture stdout from unzip process.")
        else:
            for line in process.stdout:
                # 去除首尾空白，包括換行符
                clean_line = line.strip()
                if clean_line:
                    # 打印當前正在解壓縮的檔案
                    print(f"\r\033[2K extract --> {clean_line}", end="", flush=True)
                print()
        # 5. 等待子進程完成並獲取返回碼
        process.wait()
        # 6. 檢查返回碼以確定是否成功
        if process.returncode == 0:
            logger.info("-" * 30)
            logger.info("Extraction completed successfully, all files saved to %s.", extract_dir)
            return extract_file_diff(original_files, extract_dir)
        else:
            logger.error("-" * 30)
            # 如果失敗，通常錯誤訊息已在 stdout/stderr 中輸出
            logger.error("unzip command failed with return code %d", process.returncode)
    except FileNotFoundError:
        logger.error("unzip command not found. Please ensure unzip is installed on your system.")
    except Exception as e:
        logger.error("Unknown error occurred while executing subprocess: %s", e)
