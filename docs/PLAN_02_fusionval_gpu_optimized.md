# Plan 02: Fusion-Val r Selection and GPU Throughput Optimization

## Summary

The main issue in the first exploratory experiment was not only low CUDA utilization. The more serious issue was experimental leakage: selecting `best r` on the test set makes the result invalid as formal evidence.

This plan fixes the experiment protocol first, then improves GPU throughput with controlled multi-process execution.

Default goal: balanced progress between scientific rigor and runtime efficiency.

Default parallel strategy: controlled multi-process concurrency with VRAM gating, independent logs, and resumable run directories.

## Key Changes

### Three-Way Split

Use CIFAR-10 train data for three non-overlapping subsets:

- `model_train`: 4000 images per class, used to train the five classifiers.
- `checkpoint_val`: 500 images per class, used for model checkpoint selection.
- `fusion_val`: 500 images per class, used for selecting `r`.
- `test`: CIFAR-10 original test images for the selected CIFAR-5 classes, used only for final reporting.

`per_model_train_size` is 4000. This makes `class-skewed-private + overlap=0` strictly feasible and prevents the last model from receiving extra samples.

### Correct Analysis Protocol

Formal result:

- Select `r` on `fusion_val`.
- Report accuracy, NLL, Brier, and ECE on `test`.

Diagnostic only:

- Keep `oracle_test_best_r.csv` to understand the gap between fusion-val selection and test-optimal selection.
- Do not report `oracle_test_best_r` as a formal result.

Required output files:

- `selected_r_by_fusion_val.csv`
- `test_metrics_by_selected_r.csv`
- `oracle_test_best_r.csv`
- `diversity_summary.csv`
- `per_class_selected_r.csv`
- `aggregate_test_metrics_by_condition.csv`
- `selected_r_frequency.csv`
- `diversity_metric_correlations.csv`

### Statistical Reliability

Default seeds:

- `2026`
- `2027`
- `2028`

Experiment grid:

- 2 partition modes.
- 5 overlap ratios.
- 3 seeds.

Report:

- Mean.
- Standard deviation.
- 95% confidence interval.
- Relationship between overlap and selected `r`.
- Relationship between partition mode and selected `r`.
- Relationship between diversity metrics and selected `r` or test accuracy.

### GPU Throughput Optimization

Add controlled runner options:

- `--max-concurrent-runs`
- `--parallel-models`
- `--batch-size`
- `--amp`
- `--disable-progress`
- `--prefetch-factor`
- `--compile`

Initial recommended configuration:

- `max_concurrent_runs=2`
- `parallel_models=2`
- `batch_size=512`
- `amp=auto`

If peak VRAM stays low and no OOM occurs, benchmark a more aggressive setting. If OOM or DataLoader contention appears, fall back to:

- `max_concurrent_runs=2`
- `parallel_models=1`

### Training Script Performance Settings

Implemented:

- AMP with `auto|none|fp16|bf16`.
- EDL loss internal `alpha.float()` for stable `log` and `digamma`.
- `non_blocking=True` for CUDA transfers.
- `persistent_workers=True` when `num_workers>0`.
- `prefetch_factor=2`.
- `drop_last=True` by default for training loaders.
- `torch.backends.cudnn.benchmark=True`.
- `torch.set_float32_matmul_precision("high")`.
- `--disable-progress` for clean multi-process logs.
- Optional `--compile`, not enabled by default.

## Benchmark Plan

Run a short 3-epoch benchmark on:

- partition mode: `stratified-balanced`
- overlap: `0.5`
- seed: `2026`, plus `2027` where needed for two-process cases

Benchmark configurations:

- A: single process, `parallel_models=5`, `batch_size=512`.
- B: two processes, `max_concurrent_runs=2`, `parallel_models=2`, `batch_size=512`.
- C: two processes, `max_concurrent_runs=2`, `parallel_models=1`, `batch_size=768`.

Record:

- Effective training images/sec.
- Epoch time.
- Average and max sampled GPU utilization.
- Sampled peak GPU memory.
- Manifest peak VRAM.
- Return code.
- NaN detection.

Selection rule:

- Exclude OOM, NaN, and unstable configurations.
- Choose the highest stable throughput.
- If the fastest setting is less than 10% faster than the next best, choose the clearer and lower-risk configuration.

Observed benchmark in this workspace:

| Configuration | Effective images/sec | Average GPU util | Sampled peak memory | Decision |
|---|---:|---:|---:|---|
| A: single process, parallel_models=5, bs512 | 505.1 | 23.8% | 3373 MiB | Too slow |
| B: two processes, parallel_models=2, bs512 | 823.6 | 24.4% | 4013 MiB | Recommended |
| C: two processes, parallel_models=1, bs768 | 880.6 | 16.4% | 4398 MiB | Slightly faster, but less clean |

Final recommendation: use configuration B because C is only about 6.9% faster, below the 10% threshold, while B keeps logs and run management clearer.

## Test Plan

### Static Check

```powershell
E:\pytorch_cuda_env\python.exe -m py_compile `
  train_cifar5_edl.py `
  export_outputs.py `
  analyze_routofk_fusionval.py `
  run_routofk_overlap_experiments.py `
  run_routofk_condition.py `
  benchmark_routofk_throughput.py `
  validate_routofk_splits.py `
  edl_cifar5\data_splits.py `
  edl_cifar5\train_utils.py
```

### Split Validation

Validate every run with:

```powershell
E:\pytorch_cuda_env\python.exe validate_routofk_splits.py `
  --run-dir <run_dir> `
  --expected-model-train-per-class 4000 `
  --expected-checkpoint-val-per-class 500 `
  --expected-fusion-val-per-class 500 `
  --expected-per-model-train-size 4000
```

This checks:

- Each class has exactly 4000/500/500 samples across the three splits.
- `model_train`, `checkpoint_val`, and `fusion_val` are disjoint.
- Each model has exactly 4000 training indices.
- No duplicate indices appear within a model split.
- The actual mean pairwise overlap equals the requested overlap.

### Smoke Test

Run:

```powershell
E:\pytorch_cuda_env\python.exe run_routofk_overlap_experiments.py `
  --data-dir E:\edl_cifar5_fastai `
  --output-root runs\v2_smoke_fusionval `
  --epochs 1 `
  --batch-size 512 `
  --num-workers 2 `
  --parallel-models 2 `
  --max-concurrent-runs 1 `
  --device cuda `
  --seeds 2026 `
  --overlaps 0.0,0.5 `
  --partition-modes stratified-balanced,class-skewed-private `
  --skip-existing `
  --disable-progress
```

Smoke-test results are not formal scientific results.

### Full Experiment

```powershell
E:\pytorch_cuda_env\python.exe run_routofk_overlap_experiments.py `
  --data-dir E:\edl_cifar5_fastai `
  --output-root E:\edl_cifar5_runs\routofk_overlap_v2_fusionval `
  --epochs 30 `
  --batch-size 512 `
  --num-workers 2 `
  --parallel-models 2 `
  --max-concurrent-runs 2 `
  --device cuda `
  --seeds 2026,2027,2028 `
  --overlaps 1.0,0.75,0.5,0.25,0.0 `
  --partition-modes stratified-balanced,class-skewed-private `
  --amp auto `
  --disable-progress `
  --skip-existing
```

## Implementation Files

- `train_cifar5_edl.py`
- `export_outputs.py`
- `analyze_routofk_fusionval.py`
- `run_routofk_condition.py`
- `run_routofk_overlap_experiments.py`
- `benchmark_routofk_throughput.py`
- `validate_routofk_splits.py`
- `edl_cifar5/data_splits.py`
- `edl_cifar5/train_utils.py`
- `edl_cifar5/fusion.py`
- `edl_cifar5/diversity.py`

## Assumptions

- Keep `K=5`; do not change the ensemble size in this experiment.
- Do not switch to larger models just to increase GPU utilization.
- Do not install extra packages unless explicitly allowed.
- Do not delete previous experiment outputs.
- Treat benchmark and smoke-test metrics as engineering diagnostics, not paper results.
