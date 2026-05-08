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
    device = mask.device
    h, w = mask.shape

    # Convert to uncompressed run length encoding in GPU
    # coco tools expect fortran order (column-wise)
    with nvtx_range_if_cuda("transpose and flatten", device):
        mask_flat = mask.t().contiguous().view(-1)
    with nvtx_range_if_cuda("unique consecutive", device):
        values, lengths = torch.unique_consecutive(mask_flat, return_counts=True)
    with nvtx_range_if_cuda("counts", device):
        counts = lengths.cpu().tolist()
    with nvtx_range_if_cuda("insert 0", device):
        if values[0] == 1:
            counts.insert(0, 0)

    with nvtx_range_if_cuda("compress", device):
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
    include_dense_mask: bool = False,
) -> Generator[Tuple[torch.Tensor, dict, Optional[torch.Tensor]], None, None]:
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


    with nvtx_range_if_cuda("preprocess bboxes and strip masks", masks.device):
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

    with nvtx_range_if_cuda("letterbox mask bbox scaling", masks.device):
        new_w, new_h = _letterbox_content_size(inference_size, padding)
        num_instances, hm, wm = masks.shape
        scale_to_mask = torch.tensor(
            [wm / new_w, hm / new_h, wm / new_w, hm / new_h],
            dtype=bboxes_content_xyxy.dtype,
            device=bboxes_content_xyxy.device,
        )
        boxes_in_mask_coordinate_space = bboxes_content_xyxy * scale_to_mask

    with nvtx_range_if_cuda("floor and ceil bboxes", masks.device):
        # shape N, 4
        boxes_in_mask_coordinate_space[:, 0:2].floor_()
        boxes_in_mask_coordinate_space[:, 2:4].ceil_()
        boxes_in_mask_coordinate_space_int = boxes_in_mask_coordinate_space.long()
        boxes_in_mask_coordinate_space_int[:, 0:2].clamp_(min=0)
        boxes_in_mask_coordinate_space_int[:, 2].clamp_(max=wm)
        boxes_in_mask_coordinate_space_int[:, 3].clamp_(max=hm)
        boxes_in_mask_coordinate_space_int = boxes_in_mask_coordinate_space_int.cpu()

        boxes_in_mask_original_coordinate_space = image_bboxes[:, :4].clone()
        boxes_in_mask_original_coordinate_space[:, 0:2].floor_()
        boxes_in_mask_original_coordinate_space[:, 2:4].ceil_()
        boxes_in_mask_original_coordinate_space_int = boxes_in_mask_original_coordinate_space.long()
        boxes_in_mask_original_coordinate_space_int[:, 0:2].clamp_(min=0)
        boxes_in_mask_original_coordinate_space_int[:, 2].clamp_(max=original_size.width)
        boxes_in_mask_original_coordinate_space_int[:, 3].clamp_(max=original_size.height)
        boxes_in_mask_original_coordinate_space_int = boxes_in_mask_original_coordinate_space_int.cpu()

        torch.cuda.synchronize()

    with nvtx_range_if_cuda("create buffer", masks.device):
        buffer = torch.empty(
            (size_after_pre_processing.height, size_after_pre_processing.width),
            device=masks.device,
            dtype=torch.bool,
        )

    for i in range(num_instances):
        with nvtx_range_if_cuda("zero buffer", masks.device):
            buffer.zero_()

        with nvtx_range_if_cuda("crop mask", masks.device):
            mx1, my1, mx2, my2 = boxes_in_mask_coordinate_space_int[i]
            cropped_mask = masks[i:i+1, my1:my2 + 1, mx1:mx2 + 1].contiguous()

            ox1, oy1, ox2, oy2 = boxes_in_mask_original_coordinate_space_int[i]
            resize_target_w = ox2 - ox1
            resize_target_h = oy2 - oy1

        with nvtx_range_if_cuda("resize and binarize", masks.device):
            buffer[oy1:oy2, ox1:ox2] = (
                functional.resize(
                    cropped_mask,
                    [resize_target_h, resize_target_w],
                    interpolation=functional.InterpolationMode.BILINEAR,
                )
                .gt_(binarization_threshold)
                .to(dtype=torch.bool)
            )

        with nvtx_range_if_cuda("build rle", masks.device):
            converted = rle_build_fn(buffer, image_bboxes[i])

        with nvtx_range_if_cuda("yielding result", image_bboxes.device):
            if include_dense_mask:
                yield image_bboxes[i], converted, buffer.clone().cpu()
            else:
                yield image_bboxes[i], converted

    return None
