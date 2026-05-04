"""Run instance-segmentation inference via InferenceModelsInstanceSegmentationAdapter (ONNX)."""

from __future__ import annotations

import os
from pathlib import Path
from typing import FrozenSet

import click
from dotenv import load_dotenv

ALLOWED_MODEL_IDS: FrozenSet[str] = frozenset({
    "yolov8n-seg-640",
    "rfdetr-seg-nano",
})

# All VALID_INFERENCE_MODELS_BACKENDS values except "onnx" (see inference.core.env).
ALLOWED_BACKENDS: FrozenSet[str] = frozenset({
    "onnx",
    "torch",
    "torch-script",
    "trt",
    "hugging-face",
    "ultralytics",
    "mediapipe",
    "custom",
})

_REPO_ROOT = Path(__file__).resolve().parents[3]


def _restrict_inference_models_to_backend(backend: str) -> None:
    disabled = ALLOWED_BACKENDS - {backend}
    os.environ["DISABLED_INFERENCE_MODELS_BACKENDS"] = ",".join(sorted(disabled))


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.option(
    "--model-id",
    type=click.Choice(ALLOWED_MODEL_IDS, case_sensitive=True),
    required=True,
    help="Roboflow model id (instance segmentation).",
)
@click.option(
    "--backend",
    type=click.Choice(ALLOWED_BACKENDS, case_sensitive=True),
    default="onnx",
    show_default=True,
    help="Inference-models backend (see inference.core.env VALID_INFERENCE_MODELS_BACKENDS).",
)
@click.option(
    "--image",
    "image_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    required=True,
    help="Input image.",
)
def main(
    model_id: str,
    backend: str,
    image_path: Path,
) -> None:
    """Load weights with inference-models (ONNX), run preprocess → forward → postprocess."""
    load_dotenv(_REPO_ROOT / ".env")
    api_key = os.getenv("ROBOFLOW_API_KEY")

    if not api_key:
        raise click.ClickException(
            "ROBOFLOW_API_KEY is not set. Add it to a `.env` file at the repository "
            "root or current working directory, or export it in the environment."
        )

    _restrict_inference_models_to_backend(backend)

    # Import after backend env override (inference.core.env reads DISABLED_* at import time).
    from inference.core.models.inference_models_adapters import (
        InferenceModelsInstanceSegmentationAdapter,
    )

    model = InferenceModelsInstanceSegmentationAdapter(model_id=model_id, api_key=api_key)
    responses = model.infer(str(image_path))

    if not isinstance(responses, list):
        responses = [responses]

    total_preds = sum(len(r.predictions) for r in responses)

    click.echo(
        f"model_id={model_id} backend={backend} image={image_path}\n"
        f"batches={len(responses)} total_instance_predictions={total_preds}"
    )

    for i, r in enumerate(responses):
        click.echo(
            f"  [{i}] size={r.image.width}x{r.image.height} "
            f"instances={len(r.predictions)}"
        )


if __name__ == "__main__":
    main()
