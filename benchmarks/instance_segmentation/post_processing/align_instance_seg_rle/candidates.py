from typing import Callable, Generator, Tuple

import numpy as np
import torch
from pycocotools import mask as mask_utils
from torchvision.transforms import functional

from inference_models.entities import ImageDimensions
from inference_models.models.common.roboflow.model_packages import StaticCropOffset
from inference_models.models.common.rle_utils import torch_mask_to_coco_rle
from benchmarks.instance_segmentation.post_processing.align_instance_seg_rle.profiling import nvtx_range_if_cuda


def torch_mask_to_coco_rle_old(mask: torch.Tensor) -> dict:
    np_mask = np.asfortranarray(mask.detach().cpu().numpy().astype(np.uint8))
    return mask_utils.encode(np_mask)


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
    rle_build_fn: Callable[[torch.Tensor], dict] = torch_mask_to_coco_rle,
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

    with nvtx_range_if_cuda("padding bboxes", image_bboxes.device):
        pad_left, pad_top, pad_right, pad_bottom = padding
        offsets = torch.tensor(
            [pad_left, pad_top, pad_left, pad_top],
            device=image_bboxes.device,
        )
        image_bboxes[:, :4].sub_(offsets)

    with nvtx_range_if_cuda("scaling bboxes", image_bboxes.device):
        scale = torch.as_tensor(
            [scale_width, scale_height, scale_width, scale_height],
            dtype=image_bboxes.dtype,
            device=image_bboxes.device,
        )
        image_bboxes[:, :4].div_(scale)

    with nvtx_range_if_cuda("canvas bboxes", image_bboxes.device):
        needs_canvas = static_crop_offset.offset_x > 0 or static_crop_offset.offset_y > 0
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


    target_h = size_after_pre_processing.height
    target_w = size_after_pre_processing.width
    offset_y = static_crop_offset.offset_y
    offset_x = static_crop_offset.offset_x
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
                converted = rle_build_fn(resized[0])

        with nvtx_range_if_cuda("yielding result", image_bboxes.device):
            del resized
            yield image_bboxes[i], converted

    return None


def align_instance_segmentation_results_to_rle_masks_cropped(
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
    rle_build_fn: Callable[[torch.Tensor], dict] = torch_mask_to_coco_rle,
) -> Generator[Tuple[torch.Tensor, dict], None, None]:
    """
    Same contract as ``align_instance_segmentation_results_to_rle_masks``, but
    resizes only a tight crop around each instance bounding box (mapped into
    stripped mask space) instead of the full instance mask.

    This reduces work when boxes are small relative to the letterboxed frame.
    Bilinear resize is still applied to the crop only; values along the crop
    boundary can differ slightly from a full-frame resize followed by slicing,
    but behaviour matches when the crop spans the entire mask.

    NOTE: ``image_bboxes`` is modified in-place (same as the non-cropped
    generator). Pass a ``.clone()`` if that is not acceptable.
    """
    if image_bboxes.shape[0] == 0:
        return None

    with nvtx_range_if_cuda("padding bboxes", image_bboxes.device):
        pad_left, pad_top, pad_right, pad_bottom = padding
        offsets = torch.tensor(
            [pad_left, pad_top, pad_left, pad_top],
            device=image_bboxes.device,
        )
    
    with nvtx_range_if_cuda("scaling bboxes", image_bboxes.device):
        image_bboxes[:, :4].sub_(offsets)
        scale = torch.as_tensor(
            [scale_width, scale_height, scale_width, scale_height],
            dtype=image_bboxes.dtype,
            device=image_bboxes.device,
        )
        image_bboxes[:, :4].div_(scale)

    with nvtx_range_if_cuda("canvas bboxes", image_bboxes.device):
        needs_canvas = static_crop_offset.offset_x > 0 or static_crop_offset.offset_y > 0
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

    hm = masks.shape[1]
    wm = masks.shape[2]
    target_h = size_after_pre_processing.height
    target_w = size_after_pre_processing.width
    offset_y = static_crop_offset.offset_y
    offset_x = static_crop_offset.offset_x
    num_instances = image_bboxes.shape[0]
    for i in range(num_instances):
        with nvtx_range_if_cuda("cropping mask", image_bboxes.device):
            x1 = image_bboxes[i, 0] - offset_x
            y1 = image_bboxes[i, 1] - offset_y
            x2 = image_bboxes[i, 2] - offset_x
            y2 = image_bboxes[i, 3] - offset_y

            x1 = torch.clamp(x1, 0, target_w)
            x2 = torch.clamp(x2, 0, target_w)
            y1 = torch.clamp(y1, 0, target_h)
            y2 = torch.clamp(y2, 0, target_h)

            px1_i = int(torch.floor(x1).item())
            px2_i = int(torch.ceil(x2).item())
            py1_i = int(torch.floor(y1).item())
            py2_i = int(torch.ceil(y2).item())

            px1_i = max(0, min(px1_i, target_w - 1))
            px2_i = max(0, min(px2_i, target_w))
            py1_i = max(0, min(py1_i, target_h - 1))
            py2_i = max(0, min(py2_i, target_h))

            if px2_i <= px1_i or py2_i <= py1_i:
                if needs_canvas:
                    mask_canvas = torch.zeros(
                        (original_size.height, original_size.width),
                        dtype=torch.bool,
                        device=masks.device,
                    )
                    converted = rle_build_fn(mask_canvas)
                else:
                    converted = rle_build_fn(
                        torch.zeros(
                            (target_h, target_w),
                            dtype=torch.bool,
                            device=masks.device,
                        )
                    )
                yield image_bboxes[i], converted
                continue

        with nvtx_range_if_cuda("resize mask", image_bboxes.device):
            out_h = py2_i - py1_i
            out_w = px2_i - px1_i

            u1 = (px1_i * wm) // target_w
            u2 = (px2_i * wm + target_w - 1) // target_w
            v1 = (py1_i * hm) // target_h
            v2 = (py2_i * hm + target_h - 1) // target_h

            u1 = max(0, min(u1, wm - 1))
            u2 = max(u1 + 1, min(u2, wm))
            v1 = max(0, min(v1, hm - 1))
            v2 = max(v1 + 1, min(v2, hm))

            # we should add padding
            single = masks[i : i + 1, v1:v2, u1:u2]
            resized = (
                functional.resize(
                    single,
                    [out_h, out_w],
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
                    offset_y + py1_i : offset_y + py2_i,
                    offset_x + px1_i : offset_x + px2_i,
                ] = resized[0]
                converted = rle_build_fn(mask_canvas)
            else:
                full_mask = torch.zeros(
                    (target_h, target_w),
                    dtype=torch.bool,
                    device=resized.device,
                )
                full_mask[py1_i:py2_i, px1_i:px2_i] = resized[0]
                converted = rle_build_fn(full_mask)
        
        with nvtx_range_if_cuda("yielding result", image_bboxes.device):
            del resized
            yield image_bboxes[i], converted
    return None
