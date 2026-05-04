"""Exercise `align_instance_segmentation_results_to_rle_masks` with synthetic tensors."""

from __future__ import annotations

import torch

from inference_models.entities import ImageDimensions
from inference_models.models.common.roboflow.model_packages import StaticCropOffset
from inference_models.models.common.roboflow.post_processing import (
    align_instance_segmentation_results_to_rle_masks,
)


def letterbox_params(
    original_size: ImageDimensions, inference_size: ImageDimensions
) -> tuple[tuple[int, int, int, int], float]:
    """Match `handle_numpy_input_preparation_with_letterbox` / torch letterbox metadata."""
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
    return padding, scale


def main() -> None:
    # Original photo: 900 wide × 800 tall (ImageDimensions is height, width).
    original_size = ImageDimensions(height=800, width=900)
    inference_size = ImageDimensions(height=640, width=640)

    # Letterbox: scale = min(640/900, 640/800) → width hits 640 first; bars on top/bottom.
    # With int() resize: scaled content is 640×568, so (640−568)/2 ≈ 36 px padding
    padding, scale = letterbox_params(original_size, inference_size)
    pad_left, pad_top, pad_right, pad_bottom = padding
    print(f"padding={padding} scale={scale}")

    # Same as preprocessing: content bitmap size before pad (see PreProcessingMetadata).
    size_after_pre_processing = ImageDimensions(
        height=original_size.height,
        width=original_size.width,
    )

    # Uniform scale for both axes (Roboflow letterbox metadata).
    scale_width = scale_height = scale

    # No static crop → masks are resized to full original grid; no canvas paste.
    static_crop_offset = StaticCropOffset(
        offset_x=0,
        offset_y=0,
        crop_width=original_size.width,
        crop_height=original_size.height,
    )

    num_instances = 3
    # Per row: [x1, y1, x2, y2, confidence, class_id] in 640×640 input space (xyxy).
    # `align_instance_segmentation_results_to_rle_masks` only updates columns 0–3; 4+ are passed through.
    image_bboxes = torch.tensor(
        [
            [80.0, float(pad_top + 40), 220.0, float(pad_top + 200), 0.91, 0.0],
            [260.0, float(pad_top + 80), 420.0, float(pad_top + 320), 0.88, 1.0],
            [120.0, float(pad_top + 260), 340.0, float(pad_top + 420), 0.72, 2.0],
        ],
        dtype=torch.float32,
    )
    mask_h, mask_w = 160, 160
    masks = torch.rand(num_instances, mask_h, mask_w, dtype=torch.float32)

    bboxes = []
    rle_masks = []
    
    for bbox, mask in align_instance_segmentation_results_to_rle_masks(
        image_bboxes=image_bboxes,
        masks=masks,
        padding=padding,
        scale_width=scale_width,
        scale_height=scale_height,
        original_size=original_size,
        size_after_pre_processing=size_after_pre_processing,
        inference_size=inference_size,
        static_crop_offset=static_crop_offset,
        binarization_threshold=0.5,
    ):
        bboxes.append(bbox)
        rle_masks.append(mask)

    print(
        f"letterbox: original={original_size.width}x{original_size.height} "
        f"→ {inference_size.width}x{inference_size.height}\n"
        f"  scale={scale:.6f} padding LTRB={padding} "
        f"(top/bottom bars = {pad_top}px each)"
    )

    for i, (bbox, rle) in enumerate(zip(bboxes, rle_masks)):
        counts = rle["counts"]
        n_counts = len(counts) if hasattr(counts, "__len__") else "n/a"
        print(
            f"instance {i}: bbox[:4]={bbox[:4].tolist()} "
            f"RLE size={rle['size']} len(counts)={n_counts}"
        )


if __name__ == "__main__":
    main()
