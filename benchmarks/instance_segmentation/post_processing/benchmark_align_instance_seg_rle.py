"""Benchmark `align_instance_segmentation_results_to_rle_masks` with synthetic letterboxed data."""

from __future__ import annotations

import random
import time
from typing import List, Tuple

import click
import numpy as np
import torch

from inference_models.entities import ImageDimensions
from inference_models.models.common.roboflow.model_packages import StaticCropOffset
from inference_models.models.common.roboflow.post_processing import (
    align_instance_segmentation_results_to_rle_masks,
)


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

    box_w, box_h = 24, 24
    margin = 2
    min_x1 = pad_left + margin
    min_y1 = pad_top + margin
    max_x1 = pad_left + new_w - box_w - margin
    max_y1 = pad_top + new_h - box_h - margin
    if max_x1 < min_x1 or max_y1 < min_y1:
        raise ValueError("Letterbox content too small for the fixed box size; adjust sizes.")

    span_x = max_x1 - min_x1 + 1
    span_y = max_y1 - min_y1 + 1
    rows: List[List[float]] = []
    for i in range(n):
        step = i
        x_off = step % span_x
        y_off = (step // span_x) % span_y
        x1 = float(min_x1 + x_off)
        y1 = float(min_y1 + y_off)
        x2 = x1 + box_w
        y2 = y1 + box_h
        rows.append([x1, y1, x2, y2, 0.9, float(i % 80)])

    return torch.tensor(rows, dtype=dtype, device=device)


def _sync_if_cuda(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


def _percentiles_ms(samples: List[float]) -> Tuple[float, float, float]:
    if not samples:
        return (0.0, 0.0, 0.0)
    arr = np.asarray(samples, dtype=np.float64)
    return (
        float(np.percentile(arr, 50)),
        float(np.percentile(arr, 95)),
        float(np.percentile(arr, 99)),
    )


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.option(
    "--instances",
    "-n",
    type=int,
    required=True,
    help="Number of instance rows (boxes / masks).",
)
@click.option(
    "--warmup",
    type=int,
    default=10,
    show_default=True,
    help="Iterations to run before timing (not recorded).",
)
@click.option(
    "--iterations",
    type=int,
    default=100,
    show_default=True,
    help="Timed iterations after warmup.",
)
@click.option(
    "--device",
    type=str,
    default="cpu",
    show_default=True,
    help="Torch device, e.g. cpu or cuda.",
)
@click.option(
    "--seed",
    type=int,
    default=42,
    show_default=True,
    help="Optional RNG seed for mask noise (reproducible runs).",
)
@click.option(
    "--mask-h",
    type=int,
    default=160,
    show_default=True,
)
@click.option(
    "--mask-w",
    type=int,
    default=160,
    show_default=True,
)
def main(
    instances: int,
    warmup: int,
    iterations: int,
    device: str,
    seed: int | None,
    mask_h: int,
    mask_w: int,
) -> None:
    """Benchmark align_instance_segmentation_results_to_rle_masks (letterbox same as demo script)."""
    if instances < 1:
        raise click.BadParameter("instances must be >= 1")
    if warmup < 0 or iterations < 1:
        raise click.BadParameter("warmup must be >= 0 and iterations >= 1")

    torch_device = torch.device(device)
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch_device.type == "cuda":
            torch.cuda.manual_seed_all(seed)

    original_size = ImageDimensions(height=800, width=900)
    inference_size = ImageDimensions(height=640, width=640)
    padding, scale, new_w, new_h = letterbox_params(original_size, inference_size)
    pad_left, pad_top, _, _ = padding

    size_after_pre_processing = ImageDimensions(
        height=original_size.height,
        width=original_size.width,
    )
    scale_width = scale_height = scale
    static_crop_offset = StaticCropOffset(
        offset_x=0,
        offset_y=0,
        crop_width=original_size.width,
        crop_height=original_size.height,
    )

    bboxes_template = build_image_bboxes(
        instances, pad_left, pad_top, new_w, new_h, device=torch_device
    )
    masks_template = torch.rand(
        instances, mask_h, mask_w, dtype=torch.float32, device=torch_device
    )

    def run_once(image_bboxes: torch.Tensor, masks: torch.Tensor) -> None:
        for _, _ in align_instance_segmentation_results_to_rle_masks(
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
            pass

    for _ in range(warmup):
        image_bboxes = bboxes_template.clone()
        masks = masks_template.clone()
        run_once(image_bboxes=image_bboxes, masks=masks)
    _sync_if_cuda(torch_device)

    times_ms: List[float] = []
    for _ in range(iterations):
        image_bboxes = bboxes_template.clone()
        masks = masks_template.clone()

        _sync_if_cuda(torch_device)
        t0 = time.perf_counter()

        run_once(image_bboxes=image_bboxes, masks=masks)

        _sync_if_cuda(torch_device)
        t1 = time.perf_counter()

        times_ms.append((t1 - t0) * 1000.0)

    p50, p95, p99 = _percentiles_ms(times_ms)

    click.echo(
        f"align_instance_segmentation_results_to_rle_masks\n"
        f"  device={device}  instances={instances}  mask={mask_h}x{mask_w}\n"
        f"  letterbox: original={original_size.width}x{original_size.height} "
        f"→ {inference_size.width}x{inference_size.height}  scale={scale:.6f}\n"
        f"  padding LTRB={padding}  warmup={warmup}  iterations={iterations}\n"
        f"  latency_ms: p50={p50:.4f}  p95={p95:.4f}  p99={p99:.4f}"
    )


if __name__ == "__main__":
    main()
