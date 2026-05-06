from typing import List, Tuple

import torch
from tqdm import tqdm

from inference_models.entities import ImageDimensions

def letterbox_params(
    original_size: ImageDimensions, inference_size: ImageDimensions
) -> Tuple[Tuple[int, int, int, int], float, int, int]:
    """Same as Roboflow letterbox; returns padding, scale, and embedded content size (new_w, new_h)."""
    orig_h, orig_w = original_size.height, original_size.width
    tgt_h, tgt_w = inference_size.height, inference_size.width
    scale_w = tgt_w / orig_w
    scale_h = tgt_h / orig_h
    scale = min(scale_w, scale_h)
    new_w = int(orig_w * scale)
    new_h = int(orig_h * scale)
    pad_top = int((tgt_h - new_h) / 2)
    pad_left = int((tgt_w - new_w) / 2)
    pad_right = tgt_w - pad_left - new_w
    pad_bottom = tgt_h - pad_top - new_h
    padding = (pad_left, pad_top, pad_right, pad_bottom)
    return padding, scale, new_w, new_h


def build_image_bboxes(
    n: int,
    pad_left: int,
    pad_top: int,
    new_w: int,
    new_h: int,
    *,
    box_w: int = 24,
    box_h: int = 24,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Build `n` rows of [x1, y1, x2, y2, conf, class] in letterboxed input space, inside the
    scaled content (excluding border padding). Each instance shifts the top-left by 1px in
    a serpentine pattern so boxes stay in-bounds for large `n`.
    """
    if n < 0:
        raise ValueError("n must be non-negative")
    if box_w <= 0 or box_h <= 0:
        raise ValueError("box_w and box_h must be positive")

    margin = 2
    min_x1 = pad_left + margin
    min_y1 = pad_top + margin
    max_x1 = pad_left + new_w - box_w - margin
    max_y1 = pad_top + new_h - box_h - margin
    if max_x1 < min_x1 or max_y1 < min_y1:
        raise ValueError(
            f"Letterbox content too small for box size ({box_w}, {box_h}); adjust sizes."
        )

    span_x = max_x1 - min_x1 + 1
    span_y = max_y1 - min_y1 + 1
    rows: List[List[float]] = []
    for i in tqdm(range(n), desc="Building image bboxes"):
        step = i
        x_off = step % span_x
        y_off = (step // span_x) % span_y
        x1 = float(min_x1 + x_off)
        y1 = float(min_y1 + y_off)
        x2 = x1 + box_w
        y2 = y1 + box_h
        rows.append([x1, y1, x2, y2, 0.9, float(i % 80)])

    return torch.tensor(rows, dtype=dtype, device=device)
