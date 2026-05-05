"""Benchmark `align_instance_segmentation_results_to_rle_masks` with synthetic letterboxed data."""

from __future__ import annotations

import random
import time
from typing import Callable, Dict, Generator, List, Tuple

import click
import numpy as np
import torch
import torch.cuda.nvtx as nvtx

from inference_models.entities import ImageDimensions
from inference_models.models.common.roboflow.model_packages import StaticCropOffset
from benchmarks.instance_segmentation.post_processing.align_instance_seg_rle.candidates import (
    align_instance_segmentation_results_to_rle_masks,
    align_instance_segmentation_results_to_rle_masks_cropped,
)
from benchmarks.instance_segmentation.post_processing.align_instance_seg_rle.data import (
    build_image_bboxes,
    letterbox_params,
)
from benchmarks.instance_segmentation.post_processing.align_instance_seg_rle.candidates import (
    torch_mask_to_coco_rle,
    torch_mask_to_coco_rle_old,
)

CandidateFnType = Callable[
    [
        torch.Tensor,
        torch.Tensor,
        Tuple[int, int, int, int],
        float,
        float,
        ImageDimensions,
        ImageDimensions,
        ImageDimensions,
        StaticCropOffset,
        float,
    ],
    Generator[Tuple[torch.Tensor, dict], None, None],
]

CANDIDATE_FNS: Dict[str, CandidateFnType] = {
    "default": align_instance_segmentation_results_to_rle_masks,
    "cropped": align_instance_segmentation_results_to_rle_masks_cropped,
}

RLE_BUILD_FNS: Dict[str, Callable[[torch.Tensor], dict]] = {
    "new": torch_mask_to_coco_rle,
    "old": torch_mask_to_coco_rle_old,
}


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


@click.command(
    context_settings={"help_option_names": ["-h", "--help"]},
)
@click.option(
    "--candidate-fn",
    type=click.Choice(list(CANDIDATE_FNS.keys()), case_sensitive=True),
    default="default",
    show_default=True,
    help="Candidate function to benchmark.",
)
@click.option(
    "--rle-build-fn",
    type=click.Choice(list(RLE_BUILD_FNS.keys()), case_sensitive=True),
    default="new",
    show_default=True,
    help="RLE build function to use.",
)
@click.option(
    "--instances",
    "-n",
    type=int,
    default=100,
    show_default=True,
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
    candidate_fn: CandidateFnType,
    rle_build_fn: Callable[[torch.Tensor], dict],
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

    candidate_fn = CANDIDATE_FNS[candidate_fn]
    rle_build_fn = RLE_BUILD_FNS[rle_build_fn]

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

    print(f"Mask control sum: {masks_template.sum()=}")
    print(f"BBox control sum: {bboxes_template.sum()=}")

    def run_once(image_bboxes: torch.Tensor, masks: torch.Tensor) -> None:
        with nvtx.range("post-processing"):
            for _, _ in candidate_fn(
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
                rle_build_fn=rle_build_fn,
            ):
                pass

    print(f"Warming up {warmup} iterations...")
    for _ in range(warmup):
        image_bboxes = bboxes_template.clone()
        masks = masks_template.clone()
        run_once(image_bboxes=image_bboxes, masks=masks)
    _sync_if_cuda(torch_device)

    print(f"Timing {iterations} iterations...")
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
