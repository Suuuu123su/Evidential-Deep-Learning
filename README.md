# EDL CIFAR-5 First-Stage Implementation

This folder implements the first-stage experiment scaffold for CIFAR-5:

- CIFAR-5 is defined as a configurable 5-class subset of CIFAR-10. The default classes are `0,1,2,3,4`:
  airplane, automobile, bird, cat, deer.
- Five EDL classifiers are provided: LeNet-style CNN, small CNN, small VGG, tiny ResNet, and tiny ViT.
- Each classifier outputs Dirichlet evidence, belief mass `m({c})`, ignorance mass `m(Y)`, and plausibility `pl(c)`.
- Five fusion rules are implemented: conjunctive, disjunctive, r-out-of-K, discount-conjunctive, and average.
- Five source behavior transformations are implemented for each source `H_i={R_i,D_i,V_i,AR_i,B_i}`.

No reported metric in this folder is fabricated. Metrics are only produced by running the scripts on real CIFAR data.

## Environment Check

Run this with the Python environment that has PyTorch installed:

```powershell
python edl_cifar5\check_env.py
```

The current PATH Python on this machine may be LibreOffice's embedded Python, which is not suitable for training.

## Install Dependencies

Use your normal Python environment, then install:

```powershell
python -m pip install -r edl_cifar5\requirements.txt
```

If you already have a CUDA-enabled PyTorch environment, do not reinstall PyTorch blindly. Check it first with `check_env.py`.

## Train the Five Classifiers

```powershell
python edl_cifar5\train_cifar5_edl.py --data-dir data --output-dir runs\cifar5_edl --epochs 30 --device cuda
```

For a CPU or quick functionality check, reduce epochs and optionally use a real-data subset:

```powershell
python edl_cifar5\train_cifar5_edl.py --data-dir data --output-dir runs\cifar5_debug --epochs 1 --device cpu --subset-size 1000
```

If `--subset-size` is used, any metrics are subset metrics and must not be presented as full CIFAR-5 results.

## Export Evidence Outputs

After checkpoints exist:

```powershell
python edl_cifar5\export_outputs.py --data-dir data --checkpoint-dir runs\cifar5_edl --output runs\cifar5_edl\test_outputs.npz --split test --device cuda
```

The exported `.npz` contains real model outputs:

- `alpha`: Dirichlet parameters, shape `[N,K,C]`
- `prob`: expected class probability, shape `[N,K,C]`
- `belief`: singleton belief masses `m_i({c})`, shape `[N,K,C]`
- `uncertainty`: ignorance masses `m_i(Y)`, shape `[N,K]`
- `plausibility`: `pl_i(c)=m_i({c})+m_i(Y)`, shape `[N,K,C]`
- `labels`: remapped CIFAR-5 labels, shape `[N]`

## Run Fusion Rules

```powershell
python edl_cifar5\run_fusion.py --outputs runs\cifar5_edl\test_outputs.npz --output-json runs\cifar5_edl\fusion_metrics.json
```

This evaluates the five implemented fusion rules on real exported outputs.

## r-out-of-K Training-Overlap Study

The formal v2 protocol avoids test-set leakage. It uses:

- `model_train`: 4000 images per class from CIFAR-10 train, used to train the five classifiers.
- `checkpoint_val`: 500 images per class from CIFAR-10 train, used only for checkpoint selection.
- `fusion_val`: 500 images per class from CIFAR-10 train, used only to select `r`.
- `test`: CIFAR-10 test images for the five CIFAR-5 classes, used only for final reporting.

The default per-model train size is 4000. This is intentional: it makes
`class-skewed-private + overlap=0` feasible without giving the last model extra samples.

Run one condition:

```powershell
python train_cifar5_edl.py `
  --data-dir E:\edl_cifar5_fastai `
  --data-source imagefolder `
  --no-download `
  --output-dir E:\edl_cifar5_runs\routofk_overlap\balanced_overlap_050_seed2026 `
  --epochs 30 `
  --batch-size 512 `
  --num-workers 2 `
  --parallel-models 2 `
  --device cuda `
  --per-model-splits `
  --overlap-ratio 0.5 `
  --partition-mode stratified-balanced `
  --per-model-train-size 4000 `
  --model-train-per-class 4000 `
  --checkpoint-val-per-class 500 `
  --fusion-val-per-class 500 `
  --amp auto `
  --disable-progress
```

Supported partition modes:

- `stratified-balanced`: shared and private subsets are class-balanced.
- `class-skewed-private`: shared subset is class-balanced; each model's private subset is biased toward a different class.

Export the two formal evaluation splits:

```powershell
E:\pytorch_cuda_env\python.exe export_outputs.py `
  --data-dir E:\edl_cifar5_fastai `
  --data-source imagefolder `
  --no-download `
  --checkpoint-dir E:\edl_cifar5_runs\routofk_overlap_v2_fusionval\stratified-balanced_overlap_05_seed2026 `
  --split fusion_val `
  --output E:\edl_cifar5_runs\routofk_overlap_v2_fusionval\stratified-balanced_overlap_05_seed2026\fusion_val_outputs.npz `
  --device cuda
```

Run the complete v2 grid with controlled concurrency:

```powershell
E:\pytorch_cuda_env\python.exe run_routofk_overlap_experiments.py `
  --data-dir E:\edl_cifar5_fastai `
  --output-root E:\edl_cifar5_runs\routofk_overlap_v2_fusionval `
  --epochs 30 `
  --batch-size 512 `
  --num-workers 2 `
  --parallel-models 2 `
  --max-concurrent-runs 2 `
  --seeds 2026,2027,2028 `
  --overlaps 1.0,0.75,0.5,0.25,0.0 `
  --partition-modes stratified-balanced,class-skewed-private `
  --amp auto `
  --disable-progress `
  --skip-existing
```

The runner validates every `splits.json` before analysis. If a model has more or fewer than
4000 training indices, or if `model_train`, `checkpoint_val`, and `fusion_val` overlap, the run fails.

Analyze completed v2 runs:

```powershell
E:\pytorch_cuda_env\python.exe analyze_routofk_fusionval.py `
  --runs-root E:\edl_cifar5_runs\routofk_overlap_v2_fusionval `
  --make-plots
```

Analysis outputs:

- `selected_r_by_fusion_val.csv`: formal `r` selected on `fusion_val`.
- `test_metrics_by_selected_r.csv`: final test metrics using only the fusion-val-selected `r`.
- `oracle_test_best_r.csv`: diagnostic only; do not report it as a formal result.
- `per_class_selected_r.csv`: class-wise `r` selected on `fusion_val`.
- `diversity_summary.csv`: aggregate classifier-dependence metrics on test.
- `pairwise_diversity.csv`: pairwise disagreement/double-fault/correlation metrics.
- `aggregate_test_metrics_by_condition.csv`: seed mean/std/95% CI for test metrics.
- `selected_r_frequency.csv`: selected-`r` counts by overlap and partition mode.

Throughput benchmark:

```powershell
E:\pytorch_cuda_env\python.exe benchmark_routofk_throughput.py `
  --data-dir E:\edl_cifar5_fastai `
  --output-root E:\edl_cifar5_runs\routofk_overlap_v2_benchmark `
  --device cuda `
  --num-workers 2 `
  --amp auto
```

The benchmark writes `benchmark_summary.tsv` with runtime, effective training throughput,
sampled GPU utilization, sampled GPU memory, manifest peak VRAM, and NaN detection.
