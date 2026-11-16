# Phạm vi: Tiện ích ảnh/mask dùng chung: resize, co/giãn biên, làm mờ, và điền từ pixel lân cận
# cho prior.
"""Low-level image morphology, resizing, hole filling, and filtering primitives."""
from __future__ import annotations

from collections import deque
import numpy as np
from PIL import Image, ImageFilter

try:
    import cv2  # type: ignore
except ImportError:  # pragma: no cover - local CPU data venv fallback
    cv2 = None

try:
    from scipy import ndimage as ndi  # type: ignore
except ImportError:  # pragma: no cover - local CPU data venv fallback
    ndi = None


# Resize theo kiểu dữ liệu; dùng nearest cho mask để không tạo giá trị biên trung gian.
def resize_array(array: np.ndarray, size_hw: tuple[int, int], mode: str = "linear") -> np.ndarray:
    """Resize a 2D or 3D uint8/float32 array to `(height, width)`."""
    target_h, target_w = int(size_hw[0]), int(size_hw[1])
    if array.shape[:2] == (target_h, target_w):
        return array.copy()

    if cv2 is not None:
        interp_map = {
            "nearest": cv2.INTER_NEAREST,
            "linear": cv2.INTER_LINEAR,
            "cubic": cv2.INTER_CUBIC,
            "area": cv2.INTER_AREA,
        }
        interp = interp_map.get(mode, cv2.INTER_LINEAR)
        return cv2.resize(array, (target_w, target_h), interpolation=interp)

    resample_map = {
        "nearest": Image.Resampling.NEAREST,
        "linear": Image.Resampling.BILINEAR,
        "cubic": Image.Resampling.BICUBIC,
        "area": Image.Resampling.BOX,
    }
    resample = resample_map.get(mode, Image.Resampling.BILINEAR)
    if array.dtype == np.uint8:
        return np.asarray(Image.fromarray(array).resize((target_w, target_h), resample=resample))

    arr_f = array.astype(np.float32)
    if arr_f.ndim == 2:
        pil_f = Image.fromarray(arr_f, mode="F").resize((target_w, target_h), resample=resample)
        return np.asarray(pil_f, dtype=np.float32)

    channels = [
        np.asarray(
            Image.fromarray(arr_f[:, :, c], mode="F").resize((target_w, target_h), resample=resample),
            dtype=np.float32,
        )
        for c in range(arr_f.shape[2])
    ]
    return np.stack(channels, axis=-1)


# Mở rộng mask để tạo dải tiếp giáp hoặc vùng chỉnh sửa.
def binary_dilate(mask: np.ndarray, radius: int) -> np.ndarray:
    mask_bool = np.asarray(mask, dtype=bool)
    if radius <= 0 or not mask_bool.any():
        return mask_bool.copy()
    k = int(radius) * 2 + 1
    if cv2 is not None:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        return cv2.dilate(mask_bool.astype(np.uint8), kernel).astype(bool)
    img = Image.fromarray(mask_bool.astype(np.uint8) * 255, mode="L")
    return np.asarray(img.filter(ImageFilter.MaxFilter(k))) > 127


# Co mask để lấy lõi ổn định cách mép.
def binary_erode(mask: np.ndarray, radius: int) -> np.ndarray:
    mask_bool = np.asarray(mask, dtype=bool)
    if radius <= 0 or not mask_bool.any():
        return mask_bool.copy()
    r = int(radius)
    k = r * 2 + 1
    if cv2 is not None:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        return cv2.erode(mask_bool.astype(np.uint8), kernel, borderType=cv2.BORDER_CONSTANT, borderValue=0).astype(bool)
    padded = np.pad(mask_bool.astype(np.uint8) * 255, pad_width=r, mode="constant", constant_values=0)
    eroded = np.asarray(Image.fromarray(padded, mode="L").filter(ImageFilter.MinFilter(k)))
    return eroded[r:-r, r:-r] > 127


# Lấy đường viền mask để đánh dấu mục tiêu trong crop ngữ cảnh.
def mask_outline(mask: np.ndarray, thickness: int = 2) -> np.ndarray:
    m = np.asarray(mask, dtype=bool)
    return binary_dilate(m, max(1, thickness)) & (~m)


# Lan giá trị từ pixel hợp lệ gần nhất để tạo nền khởi tạo.
def nearest_foreground_fill(rgb: np.ndarray, known_mask: np.ndarray) -> np.ndarray:
    """Extrapolate `rgb` from `known_mask` to `~known_mask` via nearest pixel."""
    known = np.asarray(known_mask, dtype=bool)
    out = rgb.copy()
    if not known.any() or known.all():
        return out

    if ndi is not None:
        _, indices = ndi.distance_transform_edt(~known, return_indices=True)
        out[~known] = rgb[indices[0][~known], indices[1][~known]]
        return out

    filled = known.copy()
    for _ in range(max(rgb.shape[:2])):
        if filled.all():
            break
        prev = filled.copy()
        for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)):
            shifted_mask = np.zeros_like(filled)
            shifted_rgb = np.zeros_like(out)
            sy0, sy1 = max(0, -dy), filled.shape[0] - max(0, dy)
            dy0, dy1 = max(0, dy), filled.shape[0] - max(0, -dy)
            sx0, sx1 = max(0, -dx), filled.shape[1] - max(0, dx)
            dx0, dx1 = max(0, dx), filled.shape[1] - max(0, -dx)
            shifted_mask[dy0:dy1, dx0:dx1] = prev[sy0:sy1, sx0:sx1]
            shifted_rgb[dy0:dy1, dx0:dx1] = out[sy0:sy1, sx0:sx1]
            newly = shifted_mask & (~filled)
            if newly.any():
                out[newly] = shifted_rgb[newly]
                filled[newly] = True
    return out


# Điền lỗ nhỏ theo màu lân cận trong bước chuẩn bị prior, trước SD1.5.
def inpaint_holes(rgb: np.ndarray, hole_mask: np.ndarray, radius: int = 7) -> np.ndarray:
    """Inpaint `hole_mask` smoothly from surrounding valid pixels."""
    holes = np.asarray(hole_mask, dtype=bool)
    if not holes.any():
        return rgb.copy()
    if cv2 is not None:
        return cv2.inpaint(rgb, holes.astype(np.uint8) * 255, float(radius), cv2.INPAINT_TELEA)
    filled = nearest_foreground_fill(rgb, ~holes)
    blurred = np.asarray(Image.fromarray(filled).filter(ImageFilter.GaussianBlur(radius=2)))
    out = rgb.copy()
    out[holes] = blurred[holes]
    return out


# Làm mờ RGB để tách thành phần màu và cấu trúc tần số thấp.
def gaussian_blur_rgb(rgb: np.ndarray, sigma: float = 4.0) -> np.ndarray:
    """Return float32 Gaussian-blurred RGB image."""
    if cv2 is not None:
        return cv2.GaussianBlur(rgb.astype(np.float32), (0, 0), sigmaX=float(sigma), sigmaY=float(sigma))
    pil_blurred = Image.fromarray(rgb.astype(np.uint8)).filter(ImageFilter.GaussianBlur(radius=float(sigma)))
    return np.asarray(pil_blurred, dtype=np.float32)
