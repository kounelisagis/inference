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


def build_synthetic_instance_masks(
    bboxes: torch.Tensor,
    mask_h: int,
    mask_w: int,
    inference_size: ImageDimensions,
) -> torch.Tensor:
    """
    Build instance masks that mimic center-heavy objectness with soft edges.

    Each mask contains:
    - a gaussian centered at the corresponding bbox center
    - spread proportional to bbox dimensions
    - light additive noise so boundary values can cross the threshold
    """
    device = bboxes.device
    dtype = torch.float32
    instances = bboxes.shape[0]

    yy = torch.arange(mask_h, device=device, dtype=dtype).view(1, mask_h, 1)
    xx = torch.arange(mask_w, device=device, dtype=dtype).view(1, 1, mask_w)

    scale_x = mask_w / inference_size.width
    scale_y = mask_h / inference_size.height

    centers_x = ((bboxes[:, 0] + bboxes[:, 2]) * 0.5) * scale_x
    centers_y = ((bboxes[:, 1] + bboxes[:, 3]) * 0.5) * scale_y

    box_w = (bboxes[:, 2] - bboxes[:, 0]).clamp_min(1.0) * scale_x
    box_h = (bboxes[:, 3] - bboxes[:, 1]).clamp_min(1.0) * scale_y

    sigma_x = (box_w * 0.28).clamp_min(1.0).view(instances, 1, 1)
    sigma_y = (box_h * 0.28).clamp_min(1.0).view(instances, 1, 1)
    mu_x = centers_x.view(instances, 1, 1)
    mu_y = centers_y.view(instances, 1, 1)

    gaussian = torch.exp(
        -(
            ((xx - mu_x) ** 2) / (2.0 * sigma_x**2)
            + ((yy - mu_y) ** 2) / (2.0 * sigma_y**2)
        )
    )

    # Keep mostly true positives inside the projected bbox while preserving
    # uncertain boundaries and sparse low-level background activations.
    objectness = 0.88 * gaussian
    edge_noise = (torch.rand_like(objectness) - 0.5) * 0.18
    background_noise = torch.rand_like(objectness) * 0.08
    masks = (objectness + edge_noise + background_noise).clamp_(0.0, 1.0)
    return masks