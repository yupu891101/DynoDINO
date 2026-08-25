import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy import ndimage
from scipy.signal import fftconvolve

matplotlib.use("Agg")

INTERPOLATION_MAPPING = {"Nearest": 0, "Bilinear": 1, "BiQuadratic": 2, "Cubic": 3}
SLICE_PERCENTILE = (5, 25, 50, 75, 95)
TOP_K_OFFSET = 2


@dataclass
class CTData:
    patient_id: str
    data_root: Path
    ct_phase_0: np.ndarray
    ct_phase_1: np.ndarray
    ct_phase_2: np.ndarray

    label_phase_0: np.ndarray
    label_phase_1: np.ndarray
    label_phase_2: np.ndarray

    ref_phase_id: int | None = field(init=False, default=None)

    all_ct: list[np.ndarray] = field(init=False, repr=False)
    all_label: list[np.ndarray] = field(init=False, repr=False)

    align_result: dict[int, int] | None = field(default=None, init=False)  # {phase_id: offset}
    clip_range: tuple[int, int] = (0, 200)  # (0, 200) (-150, 250)

    def __post_init__(
        self,
    ):
        self.all_ct = [self.ct_phase_0, self.ct_phase_1, self.ct_phase_2]
        self.all_label = [self.label_phase_0, self.label_phase_1, self.label_phase_2]


def normalized_cross_correlation_fft(array1: np.ndarray, array2: np.ndarray) -> np.ndarray:
    """
    使用 FFT 快速計算兩個陣列的標準化互相關。

    數學公式 (滑動窗口版本):
    NCC(u, v) = Σ_{x,y} [ (I(x,y) - μ_I(u,v)) * (T(x-u, y-v) - μ_T) ] /
                sqrt[ Σ_{x,y}(I(x,y) - μ_I(u,v))² * Σ_{x,y}(T(x-u, y-v) - μ_T)² ]
    其中：
    - I: 影像 (image)
    - T: 樣板 (template)
    - μ_T: 樣板的平均值
    - μ_I(u,v): 在 (u,v) 位置對應樣板大小的影像區塊的平均值

    此實現使用 FFT 進行卷積加速。

    :param array1: 主陣列 (image)
    :param array2: 樣板陣列 (template)
    :return: 正規化互相關結果陣列
    """

    array1 = array1 - array1.mean()
    array2 = array2 - array2.mean()

    cross_correlation = fftconvolve(array1, np.flip(array2), mode="same")

    array1_variance_sum = fftconvolve(array1**2, np.ones_like(array2), mode="same")
    array2_variance_sum = np.sum(array2**2)

    denominator = np.sqrt(array1_variance_sum * array2_variance_sum)
    denominator[denominator == 0] = 1e-12

    normalized_result = cross_correlation / denominator

    return normalized_result


def ncc_gpu(template: torch.Tensor, image_batch: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    使用 PyTorch FFT (torch.fft) 在 GPU 上快速計算 Batch ZNCC/NCC。
    此方法同樣採用全局零均值化 + 局部/全局混合標準化，以匹配 FFT 範式。

    數學公式 (批次滑動窗口版本):
    ZNCC(u, v) = Σ_{x,y} [ (I(x,y) - μ_I(u,v)) * (T(x-u, y-v) - μ_T) ] /
                 sqrt[ Σ_{x,y}(I(x,y) - μ_I(u,v))² * Σ_{x,y}(T(x-u, y-v) - μ_T)² ]
    其中：
    - I: 影像批次 (image batch)
    - T: 樣板 (template)
    - μ_T: 樣板的平均值
    - μ_I(u,v): 在 (u,v) 位置對應樣板大小的影像區塊的平均值
    - ZNCC (Zero-mean Normalized Cross-Correlation) 表示在計算前會對影像和樣板進行零均值化。
    此實現使用 PyTorch 的 FFT 進行卷積加速和批次處理。

    Args:
        template (Tensor): 樣板張量 (H_t, W_t)
        image_batch (Tensor): 批次圖像張量 (B, H_i, W_i)
    """
    # 1. 確保張量是四維：(B, C, H, W) 或 (1, 1, H, W)
    template = template.unsqueeze(0).unsqueeze(0)
    image_batch = image_batch.unsqueeze(1)
    _, _, H, W = image_batch.shape

    # 2. 全局零均值化
    T_centered = template - template.mean()
    I_centered = image_batch - image_batch.mean(dim=[-2, -1], keepdim=True)  # 批次圖像的全局均值

    # --- 3. FFT 步驟 ---

    # 3.1. 圖像和樣板的頻域表示 (零填充到所需的卷積尺寸)
    # 使用 I 和 T 的最大尺寸作為 FFT 尺寸 (此處假設 I 和 T 已經是相同尺寸或 I 更大)
    fft_dims = (H, W)

    I_fft = torch.fft.rfft2(I_centered, s=fft_dims)
    T_fft = torch.fft.rfft2(T_centered, s=fft_dims)

    # 3.2. 分子：互相關 = 點乘 (T_fft 的共軛) * (I_fft)
    # 注意: torch.fft.rfft2 的結果是複數
    corr_fft = I_fft * T_fft.conj()

    # 3.3. 逆 FFT 得到互相關 (分子)
    cross_correlation = torch.fft.irfft2(corr_fft, s=fft_dims)
    T_std_sum_sq = torch.sum(T_centered**2)

    I_sq = I_centered**2
    ones_kernel = torch.ones_like(T_centered)
    I_sq_fft = torch.fft.rfft2(I_sq, s=fft_dims)
    ones_fft = torch.fft.rfft2(ones_kernel, s=fft_dims)

    I_local_sq_sum = torch.fft.irfft2(I_sq_fft * ones_fft, s=fft_dims)
    denominator = torch.sqrt(I_local_sq_sum * T_std_sum_sq + eps)

    # --- 5. NCC ---
    ncc_result = cross_correlation / denominator

    # 返回 (B, H, W)
    return ncc_result.squeeze(1)


def normalized_cross_correlation_naive(image: np.ndarray, template: np.ndarray) -> float:
    """
    使用基本定義（非 FFT）計算兩個陣列的標準化互相關。
    此版本主要用於教學或驗證，速度較慢。

    數學公式：
    NCC = Σᵢ [ (Iᵢ - μᵢ) * (Tᵢ - μₜ) ] / sqrt[ Σᵢ(Iᵢ - μᵢ)² * Σᵢ(Tᵢ - μₜ)² ]
    其中：
    - I: 影像區塊
    - T: 樣板
    - μᵢ: 影像區塊的平均值
    - μₜ: 樣板的平均值
    - Σᵢ: 對區塊/樣板中的所有像素進行加總

    :param image: 來源影像區塊 (應與樣板大小相同)
    :param template: 樣板陣列
    :return: -1 到 1 之間的 NCC 分數
    """
    if image.shape != template.shape:
        raise ValueError("影像和樣板的大小必須相同。")

    image_mean = np.mean(image)
    template_mean = np.mean(template)

    image_centered = image - image_mean
    template_centered = template - template_mean

    numerator = np.sum(image_centered * template_centered)
    denominator = np.sqrt(np.sum(image_centered**2) * np.sum(template_centered**2))

    if denominator == 0:
        return 0.0  # 避免除以零

    return numerator / denominator


def slice_resize(
    dicom_array: np.ndarray,
    target_size: int | tuple[int, int],
    interpolation: Literal["Nearest", "Bilinear", "BiQuadratic", "Cubic"] = "Bilinear",
) -> np.ndarray:
    """dicom_array shape: (slice_num, H, W)"""
    if isinstance(target_size, int):
        target_h = target_w = target_size
    else:
        target_h, target_w = target_size

    _, dicom_h, dicom_w = dicom_array.shape
    resize_factor = (target_h / dicom_h, target_w / dicom_w)
    resized_dicom = np.zeros((dicom_array.shape[0], target_h, target_w))
    order = INTERPOLATION_MAPPING[interpolation]
    for idx, dicom_slice in enumerate(dicom_array):
        resized_dicom[idx, :, :] = ndimage.zoom(dicom_slice, resize_factor, order=order, prefilter=False)
    return resized_dicom


def ct_preprocess(ct_data: CTData):
    ct_arrays: list[np.ndarray] = []
    for ct_array in ct_data.all_ct:
        ct_array = (
            (ct_array.clip(*ct_data.clip_range) - min(ct_data.clip_range))
            / (max(ct_data.clip_range) - min(ct_data.clip_range))
        ).astype(np.float32)
        ct_arrays.append(ct_array)
    return ct_arrays


def cal_similarity(ref_slice: np.ndarray, target_array: np.ndarray) -> np.ndarray:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ref_tensor = None
    target_batch = None
    try:
        with torch.no_grad():
            ref_tensor = torch.from_numpy(ref_slice).to(device)
            target_batch = torch.from_numpy(target_array).to(device)
            correlation_batch = ncc_gpu(ref_tensor, target_batch)
            score, _ = correlation_batch.flatten(start_dim=1).max(dim=1)
            return score.cpu().numpy()
    except Exception as e:
        del ref_tensor, target_batch
        torch.cuda.empty_cache()
        raise e


def get_confidence(scores: np.ndarray) -> float:
    max_score = scores.max()

    # 創建一個不包含最高分的 numpy 陣列
    other_scores = scores[scores < max_score]
    average_of_others = np.mean(other_scores)
    std_of_others = np.std(other_scores)

    # 計算標準化強度
    strength = (max_score - average_of_others) / std_of_others

    return strength


def _slice_align(slice_percentile: tuple[float | int, ...], ref_array: np.ndarray, target_array: np.ndarray) -> int:
    # import time
    offset_dict: dict[float, int] = {}
    slice_indecies = [int(percentile / 100 * len(ref_array)) for percentile in slice_percentile]
    print(f"Start to cal similarity, target_array shape: {target_array.shape}")
    for slice_index in slice_indecies:
        ref_slice = ref_array[slice_index]
        score_array = cal_similarity(ref_slice, target_array)
        # np.save(f"./{int(time.time())}.npy", np.array(score_list))
        confidence = get_confidence(score_array)

        offset_dict[confidence] = int(score_array.argmax()) - slice_index
    offset_list = [v for _, v in sorted(offset_dict.items(), reverse=True)][:TOP_K_OFFSET]
    return round(sum(offset_list) / len(offset_list))


def slice_align(ct_arrays: list[np.ndarray]):
    ref_phase_id = min(enumerate(ct_arrays), key=lambda pair: pair[1].shape[0])[0]
    ref_ct_array = ct_arrays[ref_phase_id]
    offset_result: dict[int, int] = {}
    for phase_id, target_ct_array in enumerate(ct_arrays):
        if phase_id == ref_phase_id:
            offset_result[phase_id] = 0
            continue
        offset_result[phase_id] = _slice_align(SLICE_PERCENTILE, ref_ct_array, target_ct_array)
    return ref_phase_id, offset_result


def _slice_matching(
    phase_id: int,
    ct_data: CTData,
    ref_ct_array: np.ndarray,
    target_ct_array: np.ndarray,
    debug_plot: bool = False,
    plot_save_dir: str = "score_plots",
):
    matched_ct_slices: list[np.ndarray] = []
    matched_label_slices: list[np.ndarray] = []

    current_plot_dir = os.path.join(plot_save_dir, ct_data.patient_id)
    if debug_plot:
        os.makedirs(current_plot_dir, exist_ok=True)

    if len(np.unique(target_ct_array).tolist()) == 1:
        for _ in range(ref_ct_array.shape[0]):
            matched_ct_slices.append(ct_data.all_ct[phase_id][0])
            matched_label_slices.append(ct_data.all_label[phase_id][0])
        return np.stack(matched_ct_slices, axis=0), np.stack(matched_label_slices, axis=0)

    for ref_idx, ref_slice in enumerate(ref_ct_array):
        score_array = cal_similarity(ref_slice, target_ct_array)

        max_score = score_array.max()
        max_index = score_array.argmax()
        p50 = float(np.percentile(score_array, 50))
        p75 = float(np.percentile(score_array, 75))

        # 繪圖邏輯 (僅在 debug_plot=True 時執行)
        if debug_plot:
            p25 = float(np.percentile(score_array, 25))

            # 1. 宣告畫布 (Figure) 與座標軸 (Axes)
            fig, ax = plt.subplots(figsize=(10, 5))

            # 2. 將所有 plt.plot 改為透過 ax 繪製
            ax.plot(score_array, color="#1f77b4", linewidth=1.5, label="Similarity Score")

            # 標示絕對閾值 (Threshold = 0.75)
            ax.axhline(0.75, color="red", linestyle="--", alpha=0.6, label="Hard Threshold (0.75)")

            # 標示四分位距 (IQR: 25th to 75th)
            ax.axhspan(p25, p75, color="orange", alpha=0.2, label="IQR (Q1 to Q3)")
            ax.axhline(p50, color="green", linestyle="-.", alpha=0.8, label=f"Median (Q2): {p50:.3f}")
            ax.axhline(p75, color="orange", linestyle=":", alpha=0.8, label=f"75th Percentile (Q3): {p75:.3f}")

            # 標記最高分位置
            ax.plot(max_index, max_score, "ro", markersize=8, label=f"Peak: {max_score:.3f} at slice {max_index}")

            # 3. 注意：設定標題與標籤在 ax 中要加上 set_ 前綴
            ax.set_title(f"Phase {phase_id} - Ref Slice {ref_idx} Matching Status")
            ax.set_xlabel("Target Slice Index")
            ax.set_ylabel("Score")

            ax.legend(loc="upper right", fontsize="small")
            ax.grid(True, linestyle=":", alpha=0.6)

            # 4. 影響整張畫布的操作維持用 fig
            fig.tight_layout()

            # 儲存圖片並明確關閉該畫布以釋放記憶體
            fig.savefig(os.path.join(current_plot_dir, f"phase_{phase_id}_ref_{ref_idx:03d}.png"), dpi=150)
            plt.close(fig)  # 修改：明確關閉特定的 fig 物件，更安全有效

        # 原有的判斷邏輯
        if max_score > 0.75 and (max_score - p50) > (p75 - p50) * 1.5:
            matched_ct_slices.append(ct_data.all_ct[phase_id][max_index])
            matched_label_slices.append(ct_data.all_label[phase_id][max_index])
        else:
            lower_bound = 0 if ct_data.all_ct[phase_id].dtype == np.uint8 else -1024
            matched_ct_slices.append(np.ones_like(ct_data.all_ct[phase_id][0]) * lower_bound)
            matched_label_slices.append(np.zeros_like(ct_data.all_label[phase_id][0]))

    return np.stack(matched_ct_slices, axis=0), np.stack(matched_label_slices, axis=0)


def wawtace_slice_matching(ct_data: CTData):
    ct_arrays: list[np.ndarray] = ct_preprocess(ct_data)

    ref_phase_id = 0
    ref_ct_array = ct_arrays[ref_phase_id]
    matched_ct_arrays: list[np.ndarray] = []
    matched_label_arrays: list[np.ndarray] = []
    for phase_id, target_ct_array in enumerate(ct_arrays):
        if phase_id == ref_phase_id:
            matched_ct_arrays.append(ct_data.all_ct[phase_id])
            matched_label_arrays.append(ct_data.all_label[phase_id])
            continue
        mathced_ct_array, mathced_label_array = _slice_matching(
            phase_id,
            ct_data,
            ref_ct_array,
            target_ct_array,
        )
        matched_ct_arrays.append(mathced_ct_array)
        matched_label_arrays.append(mathced_label_array)
    return matched_ct_arrays, matched_label_arrays


def simple_slice_matching(ct_data: CTData):
    """
    不進行 NCC 匹配，僅統一張數（Slice 數量）。
    以基準相位（ref_phase_id，張數最少者）為基準，對其他相位進行中心對齊。
    """
    ct_arrays: list[np.ndarray] = ct_preprocess(ct_data)

    # 選擇基準相位（原本邏輯：張數最少者為基準）
    ref_phase_id = min(enumerate(ct_arrays), key=lambda pair: pair[1].shape[0])[0]
    ref_count = ct_arrays[ref_phase_id].shape[0]
    ref_spatial_shape = ct_data.all_ct[ref_phase_id].shape[1:]

    matched_ct_arrays: list[np.ndarray] = []
    matched_label_arrays: list[np.ndarray] = []

    for phase_id, target_ct_array in enumerate(ct_arrays):
        target_count = target_ct_array.shape[0]
        original_ct = ct_data.all_ct[phase_id]
        original_label = ct_data.all_label[phase_id]

        # Ensure spatial shape consistency
        if original_ct.shape[1:] != ref_spatial_shape:
            original_ct = slice_resize(original_ct, ref_spatial_shape, interpolation="Bilinear")
            original_label = slice_resize(original_label, ref_spatial_shape, interpolation="Nearest")
            target_count = original_ct.shape[0]

        if phase_id == ref_phase_id:
            matched_ct_arrays.append(original_ct)
            matched_label_arrays.append(original_label)
            continue

        if target_count == ref_count:
            matched_ct_arrays.append(original_ct)
            matched_label_arrays.append(original_label)
        elif target_count > ref_count:
            # 目標張數較多：中心裁切
            start = (target_count - ref_count) // 2
            matched_ct_arrays.append(original_ct[start : start + ref_count])
            matched_label_arrays.append(original_label[start : start + ref_count])
        else:
            # 目標張數較少：中心補零
            diff = ref_count - target_count
            pad_before = diff // 2
            pad_after = diff - pad_before

            lower_bound = 0 if original_ct.dtype == np.uint8 else -1024

            # Pad CT
            pad_width = ((pad_before, pad_after), (0, 0), (0, 0))
            padded_ct = np.pad(original_ct, pad_width, mode="constant", constant_values=lower_bound)
            matched_ct_arrays.append(padded_ct)

            # Pad Label
            padded_label = np.pad(original_label, pad_width, mode="constant", constant_values=0)
            matched_label_arrays.append(padded_label)

    return matched_ct_arrays, matched_label_arrays


def slice_matching(ct_data: CTData):
    ct_arrays: list[np.ndarray] = ct_preprocess(ct_data)

    ref_phase_id = min(enumerate(ct_arrays), key=lambda pair: pair[1].shape[0])[0]
    ref_ct_array = ct_arrays[ref_phase_id]
    matched_ct_arrays: list[np.ndarray] = []
    matched_label_arrays: list[np.ndarray] = []
    for phase_id, target_ct_array in enumerate(ct_arrays):
        if phase_id == ref_phase_id:
            matched_ct_arrays.append(ct_data.all_ct[phase_id])
            matched_label_arrays.append(ct_data.all_label[phase_id])
            continue
        mathced_ct_array, mathced_label_array = _slice_matching(
            phase_id,
            ct_data,
            ref_ct_array,
            target_ct_array,
            True,
            "/home/pairlab/Documents/Workspace/MuDino_Multi-Phase-Fusion-Dinov3/CECT_score_plots",
        )
        matched_ct_arrays.append(mathced_ct_array)
        matched_label_arrays.append(mathced_label_array)
    return matched_ct_arrays, matched_label_arrays


def alignment_padding(
    align_result: dict[int, int],
    arrays: list[np.ndarray],
    ref_phase_id: int,
    target_instance: Literal["ct", "label"],
):
    """
    根據多個 offset 對齊並用 padding_val 補齊 ref_list 和多個 tgt_list。

    Args:
        ref_list (List): 參考列表。
        targets (Dict[str, Tuple[List, int]]): 包含多個目標列表和其對應 offset 的字典。
                                            鍵為列表名稱，值為 (列表, offset)。
        padding_val (Any): 用來補齊的數值。預設為 -999。

    Returns:
        Dict[str, List]: 包含所有補齊後列表的字典。
    """
    padding_value = 0 if target_instance == "label" else -1024

    max_offset = max(0, *list(align_result.values()))
    # 1. 計算所有array需要的總長度
    # 每個列表的總長度 = 自己的長度 + 自己的 offset + 最大 offset
    # 取所有列表中的最大長度作為最終總長度
    total_length = max(
        arrays[ref_phase_id].shape[0] + max_offset,
        *[arrays[tgt_phase_id].shape[0] + max_offset - offset for tgt_phase_id, offset in align_result.items()],
    )
    for phase_id, phase_array in enumerate(arrays):
        front_padding = max_offset if phase_id == ref_phase_id else max_offset - align_result[phase_id]
        back_padding = total_length - phase_array.shape[0] - front_padding
        padding_width = ((front_padding, back_padding), (0, 0), (0, 0))
        arrays[phase_id] = np.pad(phase_array, padding_width, mode="constant", constant_values=padding_value)
    return arrays


def save_result_to_npy(ct_data: CTData, ct_npy_folder: str, label_npy_folder: str):
    ct_folder = ct_data.data_root.joinpath(ct_npy_folder)
    ct_folder.mkdir(exist_ok=True)
    label_folder = ct_data.data_root.joinpath(label_npy_folder)
    label_folder.mkdir(exist_ok=True)
    for phase_id, ct_array in enumerate(ct_data.all_ct):
        patient_folder_path = ct_folder.joinpath(ct_data.patient_id)
        patient_folder_path.mkdir(exist_ok=True)
        for slice_id, ct_slice in enumerate(ct_array):
            save_path = patient_folder_path.joinpath(f"{phase_id:02d}_{slice_id:03d}.npy")
            np.save(save_path, ct_slice)
    for phase_id, label_array in enumerate(ct_data.all_label):
        patient_folder_path = label_folder.joinpath(ct_data.patient_id)
        patient_folder_path.mkdir(exist_ok=True)
        for slice_id, label_slice in enumerate(label_array):
            save_path = patient_folder_path.joinpath(f"{phase_id:02d}_{slice_id:03d}.npy")
            np.save(save_path, label_slice)
