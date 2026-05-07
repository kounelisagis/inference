from typing import Callable, Generator, Optional, Tuple

import numpy as np
import torch
from pycocotools import mask as mask_utils
from torchvision.transforms import functional

from inference_models.entities import ImageDimensions
from inference_models.models.common.roboflow.model_packages import StaticCropOffset
from benchmarks.instance_segmentation.post_processing.align_instance_seg_rle.profiling import nvtx_range_if_cuda


def _append_pair(pairs: list, value: bool, count: int) -> None:
    """O(1) append of ``(value, count)`` to ``pairs``, merging with the previous
    pair if values match. Zero-length appends are skipped. With consistent use,
    ``pairs`` strictly alternates between ``False`` and ``True`` runs."""
    if count == 0:
        return
    if pairs and pairs[-1][0] == value:
        previous_value, previous_count = pairs[-1]
        pairs[-1] = (previous_value, previous_count + count)
    else:
        pairs.append((value, count))


def _pairs_to_coco_counts(pairs: list) -> list:
    """Convert a strictly-alternating ``(value, count)`` list into COCO uncompressed
    counts (alternating 0-run, 1-run, 0-run, ...). A leading ``0`` is inserted when
    the first pair is a 1-run so the parity invariant holds."""
    if not pairs:
        return []
    counts: list = []
    if pairs[0][0]:
        counts.append(0)
    for _, count in pairs:
        counts.append(count)
    return counts


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


def torch_mask_to_coco_optimized_v1(
    mask: torch.Tensor, bbox: Optional[torch.Tensor] = None
) -> dict:
    # Convert to uncompressed run length encoding in GPU
    # coco tools expect fortran order (column-wise)
    with nvtx_range_if_cuda("permute_contiguous", mask.device):
        mask_flat = mask.t().contiguous().view(-1)
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


def torch_mask_to_coco_optimized_v2(
    mask: torch.Tensor, bbox: Optional[torch.Tensor] = None
) -> dict:
    h, w = mask.shape

    with nvtx_range_if_cuda("transpose", mask.device):
        x = mask.t().contiguous().view(-1)

    with nvtx_range_if_cuda("diff", mask.device):
        change = torch.ones_like(x, dtype=torch.bool)
        change[1:] = x[1:] != x[:-1]

        idx = torch.nonzero(change, as_tuple=False).flatten()

        lengths = torch.empty_like(idx)
        lengths[:-1] = idx[1:] - idx[:-1]
        lengths[-1] = x.numel() - idx[-1]

    with nvtx_range_if_cuda("d->h movement", mask.device):
        counts = lengths.cpu().tolist()

    with nvtx_range_if_cuda("insert 0", mask.device):
        if x[0].item():
            counts.insert(0, 0)

    with nvtx_range_if_cuda("compress", mask.device):
        rle = mask_utils.frPyObjects({"counts": counts, "size": [h, w]}, h, w)

    return rle


def torch_mask_to_coco_optimized_v4(
    mask: torch.Tensor,
    bbox: torch.Tensor,
) -> dict:
    """
    COCO RLE in Fortran (column-major) order. Encodes only the cropped bbox
    region on GPU (single ``unique_consecutive`` call), then walks the runs on
    the CPU once to inject:
    - leading zeros (full-zero columns ``[0, pixel_x1)`` plus ``top_pad``
      zeros before the first crop column),
    - zero padding between adjacent crop columns (``bottom_pad + top_pad``),
    - trailing zeros (``bottom_pad`` of the last crop column plus full-zero
      columns ``[pixel_x2, w)``).

    Compared to per-column encoding this avoids ``crop_width`` separate kernel
    launches (``DeviceCompactInitKernel``, ``DeviceReduceByKeyKernel``,
    ``cub::DeviceRunLengthEncode``), per-column ``cudaMemcpyAsync`` and stream
    syncs — collapsing them to one of each.

    ``bbox`` is ``xyxy`` (first four elements) expressed in the **same pixel
    grid as ``mask``** — i.e. the coordinate space of ``mask.shape``, not the
    detector's inference resolution. In ``align_instance_segmentation_results_
    to_rle_masks`` the call site at the non-canvas branch already passes
    ``image_bboxes[i]`` after ``sub_(letterbox_offsets) / scale``
    (``size_after_pre_processing`` space) and the mask resized to
    ``(size_after_pre_processing.height, size_after_pre_processing.width)``,
    so the two are aligned. Portions of ``mask`` outside the clamped integer
    bbox are assumed to be zero.
    """
    h, w = mask.shape
    x1 = float(bbox[0].item())
    y1 = float(bbox[1].item())
    x2 = float(bbox[2].item())
    y2 = float(bbox[3].item())

    pixel_x1 = max(0, min(int(np.floor(x1)), w - 1))
    pixel_x2 = max(0, min(int(np.ceil(x2)), w))
    pixel_y1 = max(0, min(int(np.floor(y1)), h - 1))
    pixel_y2 = max(0, min(int(np.ceil(y2)), h))

    if pixel_x2 <= pixel_x1 or pixel_y2 <= pixel_y1:
        with nvtx_range_if_cuda("degenerate mask-bbox", mask.device):
            mask_flat = mask.t().contiguous().view(-1)
            values, lengths = torch.unique_consecutive(mask_flat, return_counts=True)
            counts = lengths.cpu().tolist()
            if values[0]:
                counts.insert(0, 0)
            return mask_utils.frPyObjects({"counts": counts, "size": [h, w]}, h, w)

    top_pad = pixel_y1
    bottom_pad = h - pixel_y2
    bbox_height = pixel_y2 - pixel_y1
    crop_width = pixel_x2 - pixel_x1
    leading_zeros = pixel_x1 * h + top_pad
    trailing_zeros = bottom_pad + (w - pixel_x2) * h
    between_columns_pad = top_pad + bottom_pad
    crop_flat_length = bbox_height * crop_width

    with nvtx_range_if_cuda("crop fortran flatten", mask.device):
        crop_flat = (
            mask[pixel_y1:pixel_y2, pixel_x1:pixel_x2].t().contiguous().view(-1)
        )

    with nvtx_range_if_cuda("unique consecutive (single)", mask.device):
        values, lengths = torch.unique_consecutive(crop_flat, return_counts=True)

    with nvtx_range_if_cuda("d->h", mask.device):
        values_list = values.cpu().tolist()
        lengths_list = lengths.cpu().tolist()

    with nvtx_range_if_cuda("cpu stitch", mask.device):
        pairs: list = []
        _append_pair(pairs, False, leading_zeros)

        position = 0
        for value, length in zip(values_list, lengths_list):
            is_one = bool(value)
            remaining = length
            while remaining > 0:
                offset_within_column = position % bbox_height
                space_in_current_column = bbox_height - offset_within_column
                chunk = (
                    remaining
                    if remaining < space_in_current_column
                    else space_in_current_column
                )
                _append_pair(pairs, is_one, chunk)
                position += chunk
                remaining -= chunk
                at_column_boundary = (position % bbox_height) == 0
                is_last_column_end = position == crop_flat_length
                if (
                    at_column_boundary
                    and not is_last_column_end
                    and between_columns_pad > 0
                ):
                    _append_pair(pairs, False, between_columns_pad)

        _append_pair(pairs, False, trailing_zeros)
        counts = _pairs_to_coco_counts(pairs)

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
                converted = rle_build_fn(resized[0], image_bboxes[i])

        with nvtx_range_if_cuda("yielding result", image_bboxes.device):
            del resized
            yield image_bboxes[i], converted

    return None
