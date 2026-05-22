#!/usr/bin/env python3
"""Micro-benchmark RF-DETR ``pre_process_network_input`` (PIL vs Triton fast path).

Isolated preprocess timing for regression tracking on a specific machine (T4, L4,
Jetson, etc.). Compare relative speedup on the same host; do not use absolute ms
across platforms.

Example::

    cd inference_models
    uv run python development/benchmark_rfdetr_preprocess.py --output /tmp/rfdetr_preprocess.json

    # PIL only / Triton only
    uv run python development/benchmark_rfdetr_preprocess.py --modes pil --output pil.json
    USE_TRITON_FOR_PREPROCESSING=1 uv run python development/benchmark_rfdetr_preprocess.py --modes triton

Requires CUDA (default). Use ``--allow-cpu`` only for smoke runs (PIL path on CPU).
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, List, Literal, Optional, Sequence, Tuple

import numpy as np
import torch

# Imports after argparse/env are not required; we patch the flag at runtime.
import inference_models.models.rfdetr.pre_processing as rfdetr_pre_processing
from inference_models.models.common.roboflow.model_packages import (
    ColorMode,
    ImagePreProcessing,
    NetworkInputDefinition,
    ResizeMode,
    TrainingInputSize,
)
from inference_models.models.rfdetr.pre_processing import pre_process_network_input

BenchmarkMode = Literal["pil", "triton"]

_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass(frozen=True)
class BenchmarkCase:
    name: str
    src_height: int
    src_width: int
    target_height: int
    target_width: int
    batch_size: int = 1
    input_color_format: str = "rgb"


DEFAULT_CASES: Tuple[BenchmarkCase, ...] = (
    BenchmarkCase(
        name="stretch_1080p_to_560",
        src_height=1080,
        src_width=1920,
        target_height=560,
        target_width=560,
    ),
    BenchmarkCase(
        name="stretch_720p_to_560",
        src_height=720,
        src_width=1280,
        target_height=560,
        target_width=560,
    ),
    BenchmarkCase(
        name="stretch_small_to_64",
        src_height=192,
        src_width=168,
        target_height=64,
        target_width=64,
    ),
    BenchmarkCase(
        name="batch_2_stretch_1080p_to_560",
        src_height=1080,
        src_width=1920,
        target_height=560,
        target_width=560,
        batch_size=2,
    ),
)


@dataclass
class LatencyStats:
    samples_ms: int
    mean_ms: float
    min_ms: float
    p50_ms: float
    p95_ms: float
    max_ms: float
    std_ms: float


@dataclass
class CaseResult:
    name: str
    mode: BenchmarkMode
    src_shape: List[int]
    target_shape: List[int]
    batch_size: int
    input_color_format: str
    path_used: str
    triton_eligible: Optional[bool]
    latency: LatencyStats
    throughput_per_s: float


@dataclass
class BenchmarkReport:
    metadata: Dict[str, Any]
    cases: List[CaseResult] = field(default_factory=list)
    comparisons: List[Dict[str, Any]] = field(default_factory=list)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark RF-DETR preprocessing (PIL vs Triton) in isolation.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Write JSON results to this path.",
    )
    parser.add_argument(
        "--modes",
        type=str,
        default="pil,triton",
        help="Comma-separated modes to run: pil, triton.",
    )
    parser.add_argument(
        "--cases",
        type=str,
        default="default",
        help="'default' for built-in cases, or comma-separated case names.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Torch device for preprocessing (default: cuda:0).",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=20,
        help="Warm-up iterations per case/mode (not timed).",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=200,
        help="Timed iterations per case/mode.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="RNG seed for synthetic uint8 images.",
    )
    parser.add_argument(
        "--allow-cpu",
        action="store_true",
        help="Allow CPU device (Triton mode will be skipped).",
    )
    parser.add_argument(
        "--parity-check",
        action="store_true",
        help="After benchmarking, assert PIL and Triton outputs match (CUDA + Triton only).",
    )
    return parser.parse_args()


def parse_modes(raw: str) -> List[BenchmarkMode]:
    modes: List[BenchmarkMode] = []
    for part in raw.split(","):
        part = part.strip().lower()
        if not part:
            continue
        if part not in ("pil", "triton"):
            raise ValueError(f"Unknown mode {part!r}; expected pil or triton")
        modes.append(part)  # type: ignore[arg-type]
    if not modes:
        raise ValueError("At least one mode required")
    return modes


def resolve_cases(raw: str) -> List[BenchmarkCase]:
    if raw.strip().lower() == "default":
        return list(DEFAULT_CASES)
    names = {c.strip() for c in raw.split(",") if c.strip()}
    selected = [c for c in DEFAULT_CASES if c.name in names]
    missing = names - {c.name for c in selected}
    if missing:
        raise ValueError(f"Unknown case name(s): {sorted(missing)}")
    return selected


def set_triton_flag(enabled: bool) -> None:
    rfdetr_pre_processing.USE_TRITON_FOR_PREPROCESSING = enabled


def clear_preprocess_caches() -> None:
    rfdetr_pre_processing._get_resample_tables_cached.cache_clear()


def sync_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def build_network_input(case: BenchmarkCase) -> NetworkInputDefinition:
    return NetworkInputDefinition(
        training_input_size=TrainingInputSize(
            height=case.target_height,
            width=case.target_width,
        ),
        dataset_version_resize_dimensions=None,
        dynamic_spatial_size_supported=False,
        color_mode=ColorMode.RGB,
        resize_mode=ResizeMode.STRETCH_TO,
        input_channels=3,
        scaling_factor=255,
        normalization=[list(_IMAGENET_MEAN), list(_IMAGENET_STD)],
    )


def make_images(case: BenchmarkCase, seed: int) -> List[np.ndarray]:
    images: List[np.ndarray] = []
    for batch_idx in range(case.batch_size):
        rng = np.random.default_rng(seed + batch_idx)
        images.append(
            rng.integers(
                0,
                256,
                size=(case.src_height, case.src_width, 3),
                dtype=np.uint8,
            )
        )
    return images


def make_inputs(case: BenchmarkCase, images: List[np.ndarray]) -> Any:
    if case.batch_size == 1:
        return images[0]
    return images


def path_label(mode: BenchmarkMode, eligible: bool, device: torch.device) -> str:
    if mode == "triton" and eligible and device.type == "cuda":
        return "triton"
    return "pil"


def compute_stats(samples_ms: Sequence[float]) -> LatencyStats:
    arr = np.asarray(samples_ms, dtype=np.float64)
    return LatencyStats(
        samples_ms=len(arr),
        mean_ms=float(np.mean(arr)),
        min_ms=float(np.min(arr)),
        p50_ms=float(np.percentile(arr, 50)),
        p95_ms=float(np.percentile(arr, 95)),
        max_ms=float(np.max(arr)),
        std_ms=float(np.std(arr)),
    )


def benchmark_callable(
    fn: Callable[[], None],
    device: torch.device,
    warmup: int,
    iterations: int,
) -> List[float]:
    for _ in range(warmup):
        fn()
        sync_device(device)

    samples_ms: List[float] = []
    for _ in range(iterations):
        sync_device(device)
        start = time.perf_counter()
        fn()
        sync_device(device)
        end = time.perf_counter()
        samples_ms.append((end - start) * 1000.0)
    return samples_ms


def try_git_commit() -> Optional[str]:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        pass
    return None


def collect_metadata(
    device: torch.device,
    seed: int,
    warmup: int,
    iterations: int,
    modes: Sequence[BenchmarkMode],
) -> Dict[str, Any]:
    cuda_meta: Dict[str, Any] = {}
    if device.type == "cuda" and torch.cuda.is_available():
        idx = device.index if device.index is not None else torch.cuda.current_device()
        props = torch.cuda.get_device_properties(idx)
        cuda_meta = {
            "cuda_available": True,
            "cuda_device_index": idx,
            "cuda_device_name": props.name,
            "cuda_capability": f"{props.major}.{props.minor}",
            "cuda_total_memory_gb": round(props.total_memory / (1024**3), 3),
            "cuda_toolkit_version": torch.version.cuda,
        }
    else:
        cuda_meta = {"cuda_available": torch.cuda.is_available()}

    return {
        "hostname": platform.node(),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "git_commit": try_git_commit(),
        "benchmark_seed": seed,
        "warmup_iterations": warmup,
        "timed_iterations": iterations,
        "modes": list(modes),
        "use_triton_for_preprocessing_env": os.environ.get(
            "USE_TRITON_FOR_PREPROCESSING"
        ),
        "triton_package_available": rfdetr_pre_processing._TRITON_AVAILABLE,
        "device": str(device),
        **cuda_meta,
    }


def run_case_mode(
    case: BenchmarkCase,
    mode: BenchmarkMode,
    device: torch.device,
    seed: int,
    warmup: int,
    iterations: int,
) -> Optional[CaseResult]:
    images = make_images(case, seed=seed)
    inputs = make_inputs(case, images)
    image_pre_processing = ImagePreProcessing()
    network_input = build_network_input(case)

    set_triton_flag(mode == "triton")
    clear_preprocess_caches()

    image_list = images
    eligible = rfdetr_pre_processing.triton_path_eligible(
        image_list=image_list,
        image_pre_processing=image_pre_processing,
        network_input=network_input,
        target_device=device,
        pre_processing_overrides=None,
    )

    if mode == "triton" and not eligible:
        print(
            f"  skip {case.name} [triton]: not eligible "
            f"(triton_available={rfdetr_pre_processing._TRITON_AVAILABLE}, device={device})",
            file=sys.stderr,
        )
        return None

    if mode == "triton" and device.type != "cuda":
        print(f"  skip {case.name} [triton]: requires CUDA", file=sys.stderr)
        return None

    used_path = path_label(mode, eligible, device)

    def _run() -> None:
        out, _meta = pre_process_network_input(
            images=inputs,
            image_pre_processing=image_pre_processing,
            network_input=network_input,
            target_device=device,
            input_color_format=case.input_color_format,
        )
        # Prevent dead-code elimination of the output tensor.
        if out.numel() > 0:
            _ = float(out[0, 0, 0, 0].item())

    print(
        f"  run {case.name} [{mode}] "
        f"({case.src_height}x{case.src_width} -> {case.target_height}x{case.target_width}, "
        f"bs={case.batch_size}, path={used_path}) ...",
        file=sys.stderr,
    )
    samples = benchmark_callable(_run, device=device, warmup=warmup, iterations=iterations)
    stats = compute_stats(samples)
    batch_factor = case.batch_size
    throughput = 1000.0 * batch_factor / stats.mean_ms if stats.mean_ms > 0 else 0.0

    return CaseResult(
        name=case.name,
        mode=mode,
        src_shape=[case.src_height, case.src_width, 3],
        target_shape=[case.target_height, case.target_width],
        batch_size=case.batch_size,
        input_color_format=case.input_color_format,
        path_used=used_path,
        triton_eligible=eligible if mode == "triton" else None,
        latency=stats,
        throughput_per_s=round(throughput, 2),
    )


def build_comparisons(results: List[CaseResult]) -> List[Dict[str, Any]]:
    by_name: Dict[str, Dict[BenchmarkMode, CaseResult]] = {}
    for r in results:
        by_name.setdefault(r.name, {})[r.mode] = r

    comparisons: List[Dict[str, Any]] = []
    for name, modes in sorted(by_name.items()):
        pil = modes.get("pil")
        triton = modes.get("triton")
        if pil is None or triton is None:
            continue
        pil_p50 = pil.latency.p50_ms
        triton_p50 = triton.latency.p50_ms
        speedup_p50 = pil_p50 / triton_p50 if triton_p50 > 0 else None
        speedup_p95 = (
            pil.latency.p95_ms / triton.latency.p95_ms
            if triton.latency.p95_ms > 0
            else None
        )
        comparisons.append(
            {
                "name": name,
                "pil_p50_ms": pil_p50,
                "triton_p50_ms": triton_p50,
                "pil_p95_ms": pil.latency.p95_ms,
                "triton_p95_ms": triton.latency.p95_ms,
                "speedup_p50": round(speedup_p50, 3) if speedup_p50 else None,
                "speedup_p95": round(speedup_p95, 3) if speedup_p95 else None,
            }
        )
    return comparisons


def check_parity(
    cases: Sequence[BenchmarkCase],
    device: torch.device,
    seed: int,
    atol: float = 0.0,
    rtol: float = 0.0,
) -> None:
    if device.type != "cuda":
        print("parity-check skipped: requires CUDA", file=sys.stderr)
        return
    if not rfdetr_pre_processing._TRITON_AVAILABLE:
        print("parity-check skipped: Triton not available", file=sys.stderr)
        return

    image_pre_processing = ImagePreProcessing()
    for case in cases:
        images = make_images(case, seed=seed)
        inputs = make_inputs(case, images)
        network_input = build_network_input(case)

        set_triton_flag(False)
        clear_preprocess_caches()
        pil_out, _ = pre_process_network_input(
            images=inputs,
            image_pre_processing=image_pre_processing,
            network_input=network_input,
            target_device=device,
            input_color_format=case.input_color_format,
        )

        set_triton_flag(True)
        clear_preprocess_caches()
        if not rfdetr_pre_processing.triton_path_eligible(
            image_list=images if case.batch_size > 1 else [images[0]],
            image_pre_processing=image_pre_processing,
            network_input=network_input,
            target_device=device,
            pre_processing_overrides=None,
        ):
            print(f"parity-check skipped for {case.name}: not triton-eligible", file=sys.stderr)
            continue

        triton_out, _ = pre_process_network_input(
            images=inputs,
            image_pre_processing=image_pre_processing,
            network_input=network_input,
            target_device=device,
            input_color_format=case.input_color_format,
        )
        torch.testing.assert_close(pil_out, triton_out, atol=atol, rtol=rtol)
        print(f"parity-check ok: {case.name}", file=sys.stderr)


def report_to_dict(report: BenchmarkReport) -> Dict[str, Any]:
    def _convert(obj: Any) -> Any:
        if hasattr(obj, "__dataclass_fields__"):
            return {k: _convert(v) for k, v in asdict(obj).items()}
        if isinstance(obj, list):
            return [_convert(x) for x in obj]
        if isinstance(obj, dict):
            return {k: _convert(v) for k, v in obj.items()}
        return obj

    return _convert(report)


def resolve_device(device_str: str, allow_cpu: bool) -> torch.device:
    device = torch.device(device_str)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise SystemExit("CUDA requested but torch.cuda.is_available() is False")
        return device
    if device.type == "cpu" and allow_cpu:
        return device
    raise SystemExit(
        f"Device {device_str!r} is not supported. Use cuda:N or pass --allow-cpu."
    )


def main() -> None:
    args = parse_args()
    modes = parse_modes(args.modes)
    cases = resolve_cases(args.cases)
    device = resolve_device(args.device, allow_cpu=args.allow_cpu)

    # Reproducibility for CUDA (where implemented).
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    print(
        f"RF-DETR preprocess benchmark: device={device}, modes={modes}, "
        f"warmup={args.warmup}, iterations={args.iterations}, seed={args.seed}",
        file=sys.stderr,
    )

    metadata = collect_metadata(
        device=device,
        seed=args.seed,
        warmup=args.warmup,
        iterations=args.iterations,
        modes=modes,
    )
    report = BenchmarkReport(metadata=metadata)

    for mode in modes:
        print(f"\n=== mode: {mode} ===", file=sys.stderr)
        for case in cases:
            result = run_case_mode(
                case=case,
                mode=mode,
                device=device,
                seed=args.seed,
                warmup=args.warmup,
                iterations=args.iterations,
            )
            if result is not None:
                report.cases.append(result)

    report.comparisons = build_comparisons(report.cases)
    payload = report_to_dict(report)

    text = json.dumps(payload, indent=2)
    if args.output:
        out_dir = os.path.dirname(os.path.abspath(args.output))
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"\nWrote {args.output}", file=sys.stderr)
    else:
        print(text)

    if report.comparisons:
        print("\n--- speedup (pil_p50 / triton_p50) ---", file=sys.stderr)
        for c in report.comparisons:
            print(
                f"  {c['name']}: p50={c['speedup_p50']}x, p95={c['speedup_p95']}x",
                file=sys.stderr,
            )

    if args.parity_check:
        print("\n=== parity check ===", file=sys.stderr)
        check_parity(cases=cases, device=device, seed=args.seed)


if __name__ == "__main__":
    main()
