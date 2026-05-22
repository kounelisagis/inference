# RF-DETR Triton preprocessing: tests and benchmarks

Planning document for the Triton fast-path in RF-DETR preprocessing (`USE_TRITON_FOR_PREPROCESSING`). Covers correctness tests and latency benchmarking strategy. **Not yet implemented** — save and use when adding coverage.

## Branch context

Changes vs `main` (high level):

| Area | Files |
|------|-------|
| Env flag | `inference_models/configuration.py` — `USE_TRITON_FOR_PREPROCESSING` (default `false`) |
| Integration | `inference_models/models/rfdetr/pre_processing.py` — eligibility, CUDA path, static crop, CHW/HWC, batching |
| Kernel | `inference_models/models/rfdetr/triton_preprocess.py` — PIL-exact resample tables + fused Triton kernel |

Changelog claim: **numerical parity** with `PIL → F.resize(antialias=True) → F.to_tensor → F.normalize`.

Existing tests: `inference_models/tests/unit_tests/models/rfdetr/test_pre_processing.py` — PIL path on **CPU only**. No Triton coverage yet.

---

## Part 1: Correctness tests

### Recommended layout

| Area | Suggested location | GPU required |
|------|-------------------|--------------|
| Eligibility, helpers, env, cache | Extend `test_pre_processing.py` | Mostly no |
| Weight tables (CPU) | New `test_triton_preprocess.py` | No |
| Kernel I/O validation | `test_triton_preprocess.py` | Yes (or skip) |
| PIL parity | Both files, `@pytest.mark.gpu_only` | Yes |
| Optional TRT smoke | Existing TRT integration tests | Yes |

### Priority order

1. **GPU parity tests** — stretch, BGR, batch, CHW, static crop (validates changelog).
2. **`triton_path_eligible` matrix** — prevents silent wrong-path bugs.
3. **Kill switch + float-tensor exclusion** — safety.
4. **Weight table + cache** — cheap CPU tests.
5. **Kernel validation errors** — documents contract.
6. **TRT smoke** — optional, higher CI cost.

### 1. `triton_path_eligible` (unit, CPU)

Table-driven tests with `monkeypatch` on `USE_TRITON_FOR_PREPROCESSING` and `_TRITON_AVAILABLE` (patch at **`inference_models.models.rfdetr.pre_processing`**, where imported).

**Must return `False` when:**

- Flag is off (default)
- `_TRITON_AVAILABLE` is False
- `target_device` is CPU
- Contrast or grayscale enabled in `ImagePreProcessing`
- `_needs_two_step_resize(network_input)` is True (non-stretch + `dataset_version_resize_dimensions`)
- `input_channels != 3`
- `scaling_factor` not in `(None, 255)`
- `normalization is None`
- `resize_mode` not in allowed set (e.g. `FIT_LONGER_EDGE`)
- Empty `image_list`
- Unsupported inputs: float tensor, wrong `ndim`, non-uint8 numpy, wrong channel layout

**Must return `True` when** all gates pass with minimal valid config (uint8 HWC numpy, stretch, ImageNet norm, CUDA device, flag on, Triton available).

Include cases per **allowed resize mode** with `dataset_version_resize_dimensions=None` to document single-stage fast-path behavior.

### 2. Helper functions

**`looks_like_chw`** (CPU)

- Parametrize shapes: `(3, H, W)` → True; `(H, W, 3)` → False; `(1, H, W)` / `(4, H, W)`; edge cases if relevant.

**`as_hwc_uint8_cuda`** (`@pytest.mark.gpu_only` or `skipif not cuda`)

- HWC numpy → contiguous `(H, W, 3)` on CUDA
- CHW numpy → same pixels as HWC after transpose
- CHW uint8 torch → permute + device move

### 3. Env flag and import-time warning (CPU)

- `USE_TRITON_FOR_PREPROCESSING=true` + `_TRITON_AVAILABLE=False` → `pytest.warns(RuntimeWarning, match="Triton is not available")` (may need `importlib.reload` or subprocess if warning fires at import).

**Kill switch**

- On CUDA with flag **off**, `pre_process_network_input` still matches `_reference_pipeline` (mirror existing CPU scenarios). Default behavior unchanged.

### 4. Resample tables (CPU)

**`_bilinear_antialias_weights_1d_int`**

- For several `(in_size, out_size)` pairs (downscale, upscale, identity, extreme ratios):
  - `ksize`, `starts` shape, weights per output row sum ≈ `2**PRECISION_BITS` (PIL fixed-point)
  - Optional: compare against PIL/torchvision on synthetic 1D data

**`build_resample_tables`**

- Field dtypes/shapes, `ksize_y` / `ksize_x` consistent with numpy builder

**LRU cache (`get_resample_tables`)**

- Same `(device, src_h, src_w, th, tw)` → same cached object
- 51 distinct keys → bounded cache (`maxsize=50`)

### 5. `triton_preprocess_rfdetr_stretch` validation (GPU or skip)

`pytest.raises` for:

- `MissingDependencyError` when Triton unavailable (mock)
- `ModelInputError`: non-CUDA `src`, non-uint8, non-HWC-3
- `ModelRuntimeError`: wrong `out` shape/dtype/device

Happy path: optional preallocated `out` returned; shape `(1, 3, H, W)`, `float32`, CUDA.

### 6. Numerical parity (`@pytest.mark.gpu_only`)

Reuse helpers in `test_pre_processing.py`: `_reference_pipeline`, `_build_network_input`.

Pattern:

```python
# Enable flag + CUDA; compare Triton vs PIL reference
with monkeypatch ... USE_TRITON_FOR_PREPROCESSING=True:
    triton_out, meta = pre_process_network_input(..., target_device=cuda)
pil_ref = _reference_pipeline(Image.fromarray(img), ...).to(cuda)
torch.testing.assert_close(triton_out, pil_ref, atol=0, rtol=0)  # byte-exact claim
```

**Mirror existing CPU scenarios on CUDA:**

- One-step stretch, RGB uint8 numpy (several sizes: down/up/identity)
- BGR + `input_color_format="bgr"`
- `input_color_format=None` vs explicit bgr
- List batch (2+ images, different aspect ratios)
- 4D batched uint8 CHW torch (`unbind` path)
- CHW uint8 torch and CHW numpy

**Static crop (stretch, single-stage):**

- Parity vs reference on cropped region
- Metadata: `static_crop_offset`, `original_size`, `size_after_pre_processing`, scales match PIL path
- `PreProcessingOverrides(disable_static_crop=True)` → full image

**Explicit exclusions:**

- Float CHW on CUDA → must **not** use Triton; matches existing tensor branch
- Two-step letterbox → eligibility False; output still matches `test_two_step_letterbox_*` with flag on

### 7. Integration / orchestration

- Spy/mock `triton_preprocess_rfdetr_stretch`: called when eligible, not when ineligible
- Batch of N → `batch.shape[0] == N`, `len(meta) == N`
- Two images with same post-crop `(crop_h, crop_w)` → resample tables reused (call counter)

### 8. Optional heavier tests

- TRT RF-DETR smoke with `USE_TRITON_FOR_PREPROCESSING=1`
- Edge sizes: very small, extreme aspect ratio
- Fixed seed; repeated calls → identical output

### Practical test notes

- Mark CUDA tests `@pytest.mark.gpu_only`; `pytest.skip` if `not torch.cuda.is_available()` (see `writing-tests.md`).
- Patch `USE_TRITON_FOR_PREPROCESSING` on **`pre_processing`** module, not only `configuration`.
- Changelog claims byte-exact parity — prefer `atol=0, rtol=0`; document tolerance only if platform noise forces it.
- Default CI: `pytest -m "not e2e_model_inference"` — GPU parity tests need GPU runners or are skipped locally on CPU.

---

## Part 2: Benchmarking and latency regression

### Goal

Speed up preprocessing in isolation. Latency regression checks should **not** use fixed millisecond thresholds in the default unit test suite across T4 / L4 / Jetson — absolute times differ too much.

### What the repo already does

| Mechanism | Role | Hardware |
|-----------|------|----------|
| `inference_models/tests/unit_tests/...` | Correctness | Mostly CPU |
| `tests/benchmarks/core/test_speed_benchmark.py` | `pytest-benchmark` on full `model.infer` | Environment-dependent |
| `inference benchmark python-package-speed` | Published throughput (`docs/using_inference/benchmarks.md`) | L4, Jetson, etc. |
| Codeflash workflow | PR optimization (`codeflash --benchmark`) | Fixed Ubuntu setup |
| `inference_models/development/prediction/speed_test.py` | Dev script → `benchmark.json` | Local |

Full-model benchmarks **drown out** preprocess wins; use an **isolated** micro-benchmark for this path.

### Recommended split

| Layer | Purpose |
|-------|---------|
| **Unit tests** | Correctness only — no ms budgets in default CI |
| **Micro-benchmark script** | Measure PIL vs Triton; report speedup; optional baseline JSON per platform |
| **Optional slow GPU pytest** | Same-machine ratio only (`triton < pil`), not absolute ms |
| **Scheduled / labeled runners** | Automated regression vs per-platform baselines |

### Micro-benchmark tool (separate from unit tests)

Standalone script: `inference_models/development/benchmark_rfdetr_preprocess.py`

**What to measure:**

- PIL path: `USE_TRITON_FOR_PREPROCESSING=0`, `pre_process_network_input` on CUDA
- Triton path: flag `1`, same inputs/config
- Representative sizes (e.g. 1920×1080 → 560×560, one batch case)
- Warm-up iterations (resample-table cache), then timed loop
- Report p50/p95 or mean; primary metric: **speedup** = `pil / triton`

**Example output JSON:**

```json
{
  "platform": "nvidia-l4",
  "cuda_device": "NVIDIA L4",
  "driver": "...",
  "commit": "...",
  "cases": [{
    "name": "stretch_1080p_to_560",
    "pil_ms_p50": 4.2,
    "triton_ms_p50": 0.8,
    "speedup": 5.25
  }]
}
```

**Example usage:**

```bash
cd inference_models
uv run python development/benchmark_rfdetr_preprocess.py --output /tmp/rfdetr_preprocess.json
uv run python development/benchmark_rfdetr_preprocess.py --modes pil --output /tmp/pil.json
uv run python development/benchmark_rfdetr_preprocess.py --modes triton --parity-check --output /tmp/triton.json
```

Defaults: warmup=20, timed iterations=200, seed=42, compares PIL vs Triton with p50/p95 and speedup in JSON.

Document commands in PR description or this file when the script exists.

### Optional GPU pytest (ratio only)

```python
@pytest.mark.gpu_only
@pytest.mark.slow
def test_triton_preprocess_faster_than_pil_on_same_device(...):
    ...
    assert triton_p50 < pil_p50 * 0.95  # relative on same machine only
```

- Exclude from default runs: `-m "not slow"`.
- Do **not** assert `triton_p50 < 1.0` ms.
- Noisy on shared GPU CI — smoke only, not a hard gate on every PR.

### Automated regression (later)

If preprocess latency must be guarded in CI:

| Approach | Notes |
|----------|--------|
| Scheduled workflow on self-hosted runners (`gpu-l4`, `gpu-t4`, `jetson-orin`) | Real hardware |
| Committed baseline JSON per profile (`baselines/rfdetr_preprocess/l4.json`) | PR diffs speedup vs baseline |
| External metrics store | Less repo churn |

**Gate on relative metrics:**

- `speedup >= 1.5` (example), or
- `triton_p50` not worse than **10%** vs last baseline **on the same runner label**

Do **not** compare L4 baselines to Jetson.

### What not to use

- Default unit tests with fixed ms thresholds — flakes across platforms.
- `test_speed_benchmark.py` for isolated preprocess — measures full `infer`.
- Codeflash alone — optimization PRs, not a multi-platform regression matrix unless extended with stored baselines.

### Suggested implementation order (benchmarks)

1. Standalone `benchmark_rfdetr_preprocess.py` + short run instructions.
2. Optional `@pytest.mark.slow` `@pytest.mark.gpu_only` ratio test if a stable GPU runner exists.
3. Later: scheduled labeled runners + baseline JSON per `platform` tag.

---

## Part 3: Implementation checklist

### Tests (correctness)

- [ ] `test_triton_preprocess.py` — weight tables, cache, kernel validation
- [ ] Extend `test_pre_processing.py` — `triton_path_eligible`, `looks_like_chw`, kill switch, warning
- [ ] GPU parity suite — mirror CPU reference tests on CUDA with flag on
- [ ] Static crop + metadata parity on Triton path
- [ ] Optional TRT smoke with env flag

### Benchmarks (latency)

- [x] `benchmark_rfdetr_preprocess.py` script
- [ ] Document run commands (this file or contributors doc)
- [ ] Optional slow GPU ratio pytest
- [ ] Optional per-platform baseline JSON + scheduled workflow

---

## References

- Existing unit tests: `inference_models/tests/unit_tests/models/rfdetr/test_pre_processing.py`
- Preprocessing: `inference_models/inference_models/models/rfdetr/pre_processing.py`
- Kernel: `inference_models/inference_models/models/rfdetr/triton_preprocess.py`
- Env: `USE_TRITON_FOR_PREPROCESSING` in `inference_models/inference_models/configuration.py`
- Test conventions: `inference_models/docs/contributors/writing-tests.md`
- Full-model benchmarks: `docs/using_inference/benchmarks.md`
- pytest markers: `inference_models/pytest.ini` (`gpu_only`, `slow`)
