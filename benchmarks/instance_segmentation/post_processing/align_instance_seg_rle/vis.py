from typing import List, Tuple, Optional
import click
import torch
import numpy as np

from inference_models.entities import ImageDimensions


def render_masks_and_bboxes_visualization(
    results: List[Tuple[torch.Tensor, dict, Optional[torch.Tensor]]],
    sample_count: int,
    output_html: str | None,
) -> None:
    try:
        import plotly.graph_objects as go  # type: ignore[import-not-found]
        from plotly.subplots import make_subplots  # type: ignore[import-not-found]
    except ImportError as error:
        raise click.ClickException(
            "Plotly is required for visualization. Install it with `uv add plotly`."
        ) from error

    instances = len(results)
    sample_count = min(sample_count, instances)
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
        bbox, _, dense_mask = results[sample_index]
        x1, y1, x2, y2 = bbox[:4]

        fig.add_trace(
            go.Heatmap(
                z=dense_mask.cpu().numpy().astype(np.float32),
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
    fig.update_layout(title=title, height=max(420, 320 * rows))

    if output_html is not None:
        fig.write_html(output_html)
        click.echo(f"Wrote figure to {output_html}")
    else:
        fig.show()
