"""Assert equivalence between default and cropped align-to-RLE candidates."""

from __future__ import annotations

import random

import click
import numpy as np
import torch


from inference_models.entities import ImageDimensions
from inference_models.models.common.roboflow.model_packages import StaticCropOffset

from candidates import (
    align_instance_segmentation_results_to_rle_masks,
    align_instance_segmentation_results_to_rle_masks_cropped,
)
from data import (
    build_image_bboxes,
    letterbox_params,
)


def _rle_equal(left: dict, right: dict) -> bool:
    return left["size"] == right["size"] and left["counts"] == right["counts"]


@click.command(
    context_settings={
        "help_option_names": ["-h", "--help"]
    }
)
@click.option(
    "--instances",
    "-n",
    type=int,
    default=100,
    show_default=True,
)
@click.option(
    "--seed",
    type=int,
    default=42,
    show_default=True,
)
@click.option(
    "--device",
    type=str,
    default="cpu",
    show_default=True,
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
@click.option(
    "--box-h",
    type=int,
    default=24,
    show_default=True,
    help="Synthetic bbox height in letterboxed input space.",
)
@click.option(
    "--box-w",
    type=int,
    default=24,
    show_default=True,
    help="Synthetic bbox width in letterboxed input space.",
)
@click.option(
    "--strict/--no-strict",
    default=False,
    show_default=True,
    help="Raise AssertionError when any mismatch is found.",
)
def main(
    instances: int,
    seed: int,
    device: str,
    mask_h: int,
    mask_w: int,
    box_h: int,
    box_w: int,
    strict: bool,
) -> None:
    if instances < 1:
        raise click.BadParameter("instances must be >= 1")

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
    static_crop_offset = StaticCropOffset(
        offset_x=0,
        offset_y=0,
        crop_width=original_size.width,
        crop_height=original_size.height,
    )
    scale_width = scale_height = scale

    image_bboxes = build_image_bboxes(
        instances,
        pad_left,
        pad_top,
        new_w,
        new_h,
        box_w=box_w,
        box_h=box_h,
        device=torch_device,
    )
    masks = torch.rand(instances, mask_h, mask_w, dtype=torch.float32, device=torch_device)

    default_results = list(
        align_instance_segmentation_results_to_rle_masks(
            image_bboxes=image_bboxes.clone(),
            masks=masks.clone(),
            padding=padding,
            scale_width=scale_width,
            scale_height=scale_height,
            original_size=original_size,
            size_after_pre_processing=size_after_pre_processing,
            inference_size=inference_size,
            static_crop_offset=static_crop_offset,
            binarization_threshold=0.5,
        )
    )
    cropped_results = list(
        align_instance_segmentation_results_to_rle_masks_cropped(
            image_bboxes=image_bboxes.clone(),
            masks=masks.clone(),
            padding=padding,
            scale_width=scale_width,
            scale_height=scale_height,
            original_size=original_size,
            size_after_pre_processing=size_after_pre_processing,
            inference_size=inference_size,
            static_crop_offset=static_crop_offset,
            binarization_threshold=0.5,
        )
    )

    if len(default_results) != len(cropped_results):
        raise AssertionError(
            f"Length mismatch: default={len(default_results)} cropped={len(cropped_results)}"
        )

    mismatched_bbox = 0
    mismatched_rle = 0
    first_bbox_mismatch = None
    first_rle_mismatch = None
    for i, ((bbox_a, rle_a), (bbox_b, rle_b)) in enumerate(
        zip(default_results, cropped_results)
    ):
        if not torch.equal(bbox_a, bbox_b):
            mismatched_bbox += 1
            if first_bbox_mismatch is None:
                first_bbox_mismatch = i
        if not _rle_equal(rle_a, rle_b):
            mismatched_rle += 1
            if first_rle_mismatch is None:
                first_rle_mismatch = i

    same = mismatched_bbox == 0 and mismatched_rle == 0
    click.echo(
        f"comparison for {instances} instances (device={device}, mask={mask_h}x{mask_w}, box={box_w}x{box_h}, seed={seed})\n"
        f"  bbox_equal={mismatched_bbox == 0} mismatched_bbox={mismatched_bbox}\n"
        f"  rle_equal={mismatched_rle == 0} mismatched_rle={mismatched_rle}"
    )
    if first_bbox_mismatch is not None:
        click.echo(f"  first_bbox_mismatch_index={first_bbox_mismatch}")
    if first_rle_mismatch is not None:
        click.echo(f"  first_rle_mismatch_index={first_rle_mismatch}")

    if strict and not same:
        raise AssertionError("Candidate outputs are not identical")


if __name__ == "__main__":
    main()
