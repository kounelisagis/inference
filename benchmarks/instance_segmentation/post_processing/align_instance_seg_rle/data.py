from typing import List, Tuple

import click
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


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.option("--instances", type=int, default=16, show_default=True)
@click.option(
    "--sample-count",
    type=int,
    default=4,
    show_default=True,
    help="Number of first samples to visualize.",
)
@click.option("--mask-h", type=int, default=160, show_default=True)
@click.option("--mask-w", type=int, default=160, show_default=True)
@click.option("--box-h", type=int, default=24, show_default=True)
@click.option("--box-w", type=int, default=24, show_default=True)
@click.option("--original-size-h", type=int, default=800, show_default=True)
@click.option("--original-size-w", type=int, default=900, show_default=True)
@click.option("--inference-size-h", type=int, default=640, show_default=True)
@click.option("--inference-size-w", type=int, default=640, show_default=True)
@click.option("--seed", type=int, default=42, show_default=True)
@click.option(
    "--threshold",
    type=float,
    default=None,
    help="Optional mask threshold in [0, 1]. If provided, visualize binarized masks.",
)
@click.option(
    "--output-html",
    type=click.Path(dir_okay=False, writable=True),
    default=None,
    help="Optional output path for writing the Plotly figure as HTML.",
)
def main(
    instances: int,
    sample_count: int,
    mask_h: int,
    mask_w: int,
    box_h: int,
    box_w: int,
    original_size_h: int,
    original_size_w: int,
    inference_size_h: int,
    inference_size_w: int,
    seed: int,
    threshold: float | None,
    output_html: str | None,
) -> None:
    try:
        import plotly.graph_objects as go  # type: ignore[import-not-found]
        from plotly.subplots import make_subplots  # type: ignore[import-not-found]
    except ImportError as error:
        raise click.ClickException(
            "Plotly is required for visualization. Install it with `uv add plotly`."
        ) from error

    if instances < 1:
        raise click.BadParameter("instances must be >= 1")
    if sample_count < 1:
        raise click.BadParameter("sample-count must be >= 1")
    if threshold is not None and (threshold < 0.0 or threshold > 1.0):
        raise click.BadParameter("threshold must be in [0, 1]")

    torch.manual_seed(seed)
    device = torch.device("cpu")

    original_size = ImageDimensions(height=original_size_h, width=original_size_w)
    inference_size = ImageDimensions(height=inference_size_h, width=inference_size_w)
    padding, _, new_w, new_h = letterbox_params(original_size, inference_size)
    pad_left, pad_top, _, _ = padding

    bboxes = build_image_bboxes(
        instances,
        pad_left,
        pad_top,
        new_w,
        new_h,
        box_w=box_w,
        box_h=box_h,
        device=device,
    )
    masks = build_synthetic_instance_masks(
        bboxes=bboxes,
        mask_h=mask_h,
        mask_w=mask_w,
        inference_size=inference_size,
    )
    sample_count = min(sample_count, instances)
    scale_x = mask_w / inference_size.width
    scale_y = mask_h / inference_size.height
    columns = min(4, sample_count)
    rows = (sample_count + columns - 1) // columns

    fig = make_subplots(
        rows=rows,
        cols=columns,
        subplot_titles=[f"sample {i}" for i in range(sample_count)],
        horizontal_spacing=0.04,
        vertical_spacing=0.08,
    )

    for sample_index in range(sample_count):
        row = sample_index // columns + 1
        col = sample_index % columns + 1
        bbox = bboxes[sample_index]
        mask = masks[sample_index]
        display_mask = mask
        if threshold is not None:
            display_mask = mask.ge(threshold).to(dtype=torch.float32)

        x1 = float(bbox[0].item() * scale_x)
        y1 = float(bbox[1].item() * scale_y)
        x2 = float(bbox[2].item() * scale_x)
        y2 = float(bbox[3].item() * scale_y)

        fig.add_trace(
            go.Heatmap(
                z=display_mask.cpu().numpy(),
                colorscale="Viridis",
                zmin=0.0,
                zmax=1.0,
                showscale=sample_index == 0,
                colorbar={"title": "mask score"},
            ),
            row=row,
            col=col,
        )
        fig.add_shape(
            type="rect",
            x0=x1,
            y0=y1,
            x1=x2,
            y1=y2,
            line={"color": "red", "width": 2},
            fillcolor="rgba(0,0,0,0)",
            row=row,
            col=col,
        )
        fig.update_xaxes(title_text="mask x", row=row, col=col)
        fig.update_yaxes(title_text="mask y", autorange="reversed", row=row, col=col)

    title = f"First {sample_count} synthetic masks"
    if threshold is not None:
        title = f"{title} (thresholded at {threshold:.3f})"
    fig.update_layout(title=title, height=max(420, 320 * rows))

    if output_html is not None:
        fig.write_html(output_html)
        click.echo(f"Wrote figure to {output_html}")
    else:
        fig.show()


if __name__ == "__main__":
    main()
