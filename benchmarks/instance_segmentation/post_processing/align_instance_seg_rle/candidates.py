from typing import Callable, Generator, Optional, Tuple

import numpy as np
import torch
from pycocotools import mask as mask_utils
from torchvision.transforms import functional

from inference_models.entities import ImageDimensions
from inference_models.models.common.roboflow.model_packages import StaticCropOffset
from benchmarks.instance_segmentation.post_processing.align_instance_seg_rle.compact_mask import (
    CompactMask,
)
from benchmarks.instance_segmentation.post_processing.align_instance_seg_rle.profiling import (
    nvtx_range_if_cuda,
)


def _letterbox_content_size(
    inference_size: ImageDimensions, padding: Tuple[int, int, int, int]
) -> Tuple[int, int]:
    pad_left, pad_top, pad_right, pad_bottom = padding
    new_w = inference_size.width - pad_left - pad_right
    new_h = inference_size.height - pad_top - pad_bottom
    return new_w, new_h


def _preprocess_bboxes_and_strip_masks(
    image_bboxes: torch.Tensor,
    masks: torch.Tensor,
    padding: Tuple[int, int, int, int],
    scale_width: float,
    scale_height: float,
    inference_size: ImageDimensions,
    static_crop_offset: StaticCropOffset,
) -> Tuple[torch.Tensor, torch.Tensor, bool, int, int]:
    """Pad-subtract bboxes, strip letterbox padding from masks, then scale/canvas bboxes.

    Returns stripped ``masks``, ``xyxy`` in letterbox **content** coordinates (inference
    px, before scale/canvas) for mapping into mask pixels, ``needs_canvas``, and static
    crop offsets. Mutates ``image_bboxes`` in-place to final output space.
    """
    with nvtx_range_if_cuda("padding bboxes", image_bboxes.device):
        pad_left, pad_top, pad_right, pad_bottom = padding
        offsets = torch.tensor(
            [pad_left, pad_top, pad_left, pad_top],
            device=image_bboxes.device,
        )
        image_bboxes[:, :4].sub_(offsets)

    bboxes_content_xyxy = image_bboxes[:, :4].clone()

    with nvtx_range_if_cuda("padding masks", image_bboxes.device):
        n, mh, mw = masks.shape
        mask_h_scale = mh / inference_size.height
        mask_w_scale = mw / inference_size.width
        mask_pad_top, mask_pad_bottom, mask_pad_left, mask_pad_right = (
            round(mask_h_scale * pad_top),
            round(mask_h_scale * pad_bottom),
            round(mask_w_scale * pad_left),
            round(mask_w_scale * pad_right),
        )
        if (
            mask_pad_top < 0
            or mask_pad_bottom < 0
            or mask_pad_left < 0
            or mask_pad_right < 0
        ):
            masks = torch.nn.functional.pad(
                masks,
                (
                    abs(min(mask_pad_left, 0)),
                    abs(min(mask_pad_right, 0)),
                    abs(min(mask_pad_top, 0)),
                    abs(min(mask_pad_bottom, 0)),
                ),
                "constant",
                0,
            )
            padded_mask_offset_top = max(mask_pad_top, 0)
            padded_mask_offset_bottom = max(mask_pad_bottom, 0)
            padded_mask_offset_left = max(mask_pad_left, 0)
            padded_mask_offset_right = max(mask_pad_right, 0)
            masks = masks[
                :,
                padded_mask_offset_top : masks.shape[1] - padded_mask_offset_bottom,
                padded_mask_offset_left : masks.shape[2] - padded_mask_offset_right,
            ]
        else:
            masks = masks[
                :, mask_pad_top : mh - mask_pad_bottom, mask_pad_left : mw - mask_pad_right
            ]

    with nvtx_range_if_cuda("scaling bboxes", image_bboxes.device):
        scale = torch.as_tensor(
            [scale_width, scale_height, scale_width, scale_height],
            dtype=image_bboxes.dtype,
            device=image_bboxes.device,
        )
        image_bboxes[:, :4].div_(scale)

    needs_canvas = static_crop_offset.offset_x > 0 or static_crop_offset.offset_y > 0
    with nvtx_range_if_cuda("canvas bboxes", image_bboxes.device):
        if needs_canvas:
            static_crop_offsets = torch.as_tensor(
                [
                    static_crop_offset.offset_x,
                    static_crop_offset.offset_y,
                    static_crop_offset.offset_x,
                    static_crop_offset.offset_y,
                ],
                dtype=image_bboxes.dtype,
                device=image_bboxes.device,
            )
            image_bboxes[:, :4].add_(static_crop_offsets)

    return (
        masks,
        bboxes_content_xyxy,
        needs_canvas,
        static_crop_offset.offset_y,
        static_crop_offset.offset_x,
    )


def torch_mask_to_coco_rle_old(
    mask: torch.Tensor, bbox: Optional[torch.Tensor] = None
) -> dict:
    with nvtx_range_if_cuda("d->h movement", mask.device):
        np_mask = np.asfortranarray(mask.detach().cpu().numpy().astype(np.uint8))
    with nvtx_range_if_cuda("encode", mask.device):
        rle = mask_utils.encode(np_mask)
    return rle


def torch_mask_to_coco_new(
    mask: torch.Tensor, bbox: Optional[torch.Tensor] = None
) -> dict:
    # Convert to uncompressed run length encoding in GPU
    # coco tools expect fortran order (column-wise)
    with nvtx_range_if_cuda("permute", mask.device):
        mask_flat = mask.permute(1, 0).reshape(-1)
    with nvtx_range_if_cuda("unique consecutive", mask.device):
        values, lengths = torch.unique_consecutive(mask_flat, return_counts=True)
    with nvtx_range_if_cuda("counts", mask.device):
        counts = lengths.cpu().tolist()
    with nvtx_range_if_cuda("insert 0", mask.device):
        if values[0] == 1:
            counts.insert(0, 0)

    h, w = mask.shape
    with nvtx_range_if_cuda("compress", mask.device):
        rle = mask_utils.frPyObjects({"counts": counts, "size": [h, w]}, h, w)
    return rle


def align_instance_segmentation_results_to_rle_masks(
    image_bboxes: torch.Tensor,
    masks: torch.Tensor,
    padding: Tuple[int, int, int, int],
    scale_width: float,
    scale_height: float,
    original_size: ImageDimensions,
    size_after_pre_processing: ImageDimensions,
    inference_size: ImageDimensions,
    static_crop_offset: StaticCropOffset,
    binarization_threshold: float = 0.0,
    rle_build_fn: Callable[..., dict] = torch_mask_to_coco_new,
) -> Generator[Tuple[torch.Tensor, dict], None, None]:
    """
    Generator variant of align_instance_segmentation_results.

    Yields (bbox, mask) pairs one at a time. Only one full-resolution mask
    exists in memory at any given moment, so the caller can immediately
    RLE-encode it and drop the dense tensor before the next one is produced.

    NOTE: image_bboxes is modified in-place (same behaviour as the batched
    version). Pass a .clone() if that's not acceptable.
    """
    if image_bboxes.shape[0] == 0:
        return None

    masks, _, needs_canvas, offset_y, offset_x = _preprocess_bboxes_and_strip_masks(
        image_bboxes=image_bboxes,
        masks=masks,
        padding=padding,
        scale_width=scale_width,
        scale_height=scale_height,
        inference_size=inference_size,
        static_crop_offset=static_crop_offset,
    )

    target_h = size_after_pre_processing.height
    target_w = size_after_pre_processing.width
    num_instances = image_bboxes.shape[0]
    for i in range(num_instances):
        with nvtx_range_if_cuda("resizing mask", image_bboxes.device):
            # keep a batch dim so functional.resize is unambiguous
            single = masks[i : i + 1]
            resized = (
                functional.resize(
                    single,
                    [target_h, target_w],
                    interpolation=functional.InterpolationMode.BILINEAR,
                )
                .gt_(binarization_threshold)
                .to(dtype=torch.bool)
            )

        with nvtx_range_if_cuda("building rle", image_bboxes.device):
            if needs_canvas:
                mask_canvas = torch.zeros(
                    (original_size.height, original_size.width),
                    dtype=torch.bool,
                    device=resized.device,
                )
                mask_canvas[
                    offset_y : offset_y + resized.shape[1],
                    offset_x : offset_x + resized.shape[2],
                ] = resized[0]
                converted = rle_build_fn(mask_canvas)
                del mask_canvas
            else:
                converted = rle_build_fn(resized[0], image_bboxes[i])

        with nvtx_range_if_cuda("yielding result", image_bboxes.device):
            del resized
            yield image_bboxes[i], converted

    return None


def align_instance_segmentation_results_to_rle_masks_via_compact_resize(
    image_bboxes: torch.Tensor,
    masks: torch.Tensor,
    padding: Tuple[int, int, int, int],
    scale_width: float,
    scale_height: float,
    original_size: ImageDimensions,
    size_after_pre_processing: ImageDimensions,
    inference_size: ImageDimensions,
    static_crop_offset: StaticCropOffset,
    binarization_threshold: float = 0.0,
    rle_build_fn: Callable[..., dict] = torch_mask_to_coco_new,
) -> Generator[Tuple[torch.Tensor, dict], None, None]:
    """
    Like :func:`align_instance_segmentation_results_to_rle_masks`, but encodes each
    stripped mask tile as crop RLE at inference resolution, then uses
    :meth:`CompactMask.resize` (nearest-neighbour on RLE crops) to reach
    ``size_after_pre_processing``. The final COCO RLE is produced via ``rle_build_fn``
    on a dense full-frame boolean tensor (same contract as the bilinear path).

    Binarization is applied **before** compact encoding (on the small mask), unlike
    the default path which thresholds **after** bilinear upsampling.
    """
    if image_bboxes.shape[0] == 0:
        return None

    masks, bboxes_content_xyxy, needs_canvas, offset_y, offset_x = (
        _preprocess_bboxes_and_strip_masks(
            image_bboxes=image_bboxes,
            masks=masks,
            padding=padding,
            scale_width=scale_width,
            scale_height=scale_height,
            inference_size=inference_size,
            static_crop_offset=static_crop_offset,
        )
    )

    new_w, new_h = _letterbox_content_size(inference_size, padding)
    hm, wm = masks.shape[1], masks.shape[2]
    scale_to_mask = torch.tensor(
        [wm / new_w, hm / new_h, wm / new_w, hm / new_h],
        dtype=bboxes_content_xyxy.dtype,
        device=bboxes_content_xyxy.device,
    )
    bboxes_mask_xyxy = bboxes_content_xyxy * scale_to_mask

    target_h = size_after_pre_processing.height
    target_w = size_after_pre_processing.width
    num_instances = image_bboxes.shape[0]

    for i in range(num_instances):
        with nvtx_range_if_cuda("bbox d -> h", image_bboxes.device):
            xyxy_row = bboxes_mask_xyxy[i : i + 1].detach().cpu().numpy()
        with nvtx_range_if_cuda("mask d -> h", image_bboxes.device):
            mask_np = (masks[i] > binarization_threshold).detach().cpu().numpy()
        with nvtx_range_if_cuda("compact mask from dense", image_bboxes.device):
            compact = CompactMask.from_dense(
                mask_np[np.newaxis, ...],
                xyxy_row,
                (hm, wm),
            )
        with nvtx_range_if_cuda("compact mask resize", image_bboxes.device):
            compact_resized = compact.resize((target_h, target_w))
            dense_hw = compact_resized[0]

        with nvtx_range_if_cuda("dense_mask to gpu tensor", image_bboxes.device):
            full_tensor = torch.as_tensor(
                dense_hw, device=image_bboxes.device, dtype=torch.bool
            )

        with nvtx_range_if_cuda("building rle", image_bboxes.device):
            if needs_canvas:
                mask_canvas = torch.zeros(
                    (original_size.height, original_size.width),
                    dtype=torch.bool,
                    device=image_bboxes.device,
                )
                mask_canvas[
                    offset_y : offset_y + full_tensor.shape[0],
                    offset_x : offset_x + full_tensor.shape[1],
                ] = full_tensor
                converted = rle_build_fn(mask_canvas)
            else:
                converted = rle_build_fn(full_tensor, image_bboxes[i])

        with nvtx_range_if_cuda("yielding result", image_bboxes.device):
            yield image_bboxes[i], converted

    return None
