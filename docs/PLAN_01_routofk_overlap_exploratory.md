# Plan 01: r-out-of-K Training-Overlap Exploratory Study

## Status

This is the first exploratory plan for studying how the training-data overlap among five EDL classifiers affects the best `r` in an r-out-of-K fusion rule.

Important limitation: the original exploratory version selected `best r` on the test set. That is a test-set leakage risk. Results from this plan can be used for debugging and intuition only, not as formal paper evidence. The formal corrected protocol is in `PLAN_02_fusionval_gpu_optimized.md`.

## Research Question

Given five EDL classifiers trained on CIFAR-5, how do different degrees of training-data overlap and different partition modes affect the best `r` chosen by r-out-of-K fusion?

## Dataset

- Dataset: CIFAR-5, selected from CIFAR-10 classes `0,1,2,3,4`.
- Default classes: airplane, automobile, bird, cat, deer.
- Training source: CIFAR-10 train split.
- Test source: CIFAR-10 test split for the selected five classes.

## Classifier Ensemble

Train `K=5` heterogeneous EDL classifiers:

- `lenet`
- `small_cnn`
- `small_vgg`
- `tiny_resnet`
- `tiny_vit`

Each classifier outputs Dirichlet evidence. The exported outputs include:

- `alpha`: Dirichlet parameters.
- `prob`: expected class probability.
- `belief`: singleton belief masses.
- `uncertainty`: ignorance mass.
- `plausibility`: `pl(c)=m({c})+m(Y)`.

## Experimental Factors

### Training-Data Overlap

Use five overlap ratios:

- `1.0`
- `0.75`
- `0.5`
- `0.25`
- `0.0`

`overlap=1.0` means all models share the same training subset. `overlap=0.0` means no shared samples among model-specific training subsets.

### Partition Modes

- `stratified-balanced`: shared and private portions are class-balanced.
- `class-skewed-private`: shared portion is class-balanced; each model's private portion is biased toward a different class.

## Fusion Rule

Only evaluate r-out-of-K fusion in this plan.

For each class `c`, r-out-of-K estimates the support that at least `r` out of `K` classifiers support class `c`, using class-wise plausibility values.

For `K=5`, evaluate:

- `r=1`
- `r=2`
- `r=3`
- `r=4`
- `r=5`

## Metrics

For every condition and every `r`, report:

- Accuracy
- NLL
- Brier score
- ECE
- Per-class accuracy
- Pairwise diversity metrics:
  - disagreement
  - double fault
  - same wrong prediction
  - correctness correlation
  - probability correlation
  - plausibility correlation

## Intended Analysis

Analyze:

- Whether higher overlap pushes the selected `r` toward lower or higher values.
- Whether `class-skewed-private` changes the selected `r` compared with `stratified-balanced`.
- Whether classifier diversity metrics correlate with selected `r`.
- Whether r-out-of-K fusion improves over single model outputs.

## Implementation Files

Core code in this repository:

- `train_cifar5_edl.py`
- `export_outputs.py`
- `analyze_routofk_overlap.py`
- `run_fusion.py`
- `edl_cifar5/data_splits.py`
- `edl_cifar5/fusion.py`
- `edl_cifar5/diversity.py`

The improved runner `run_routofk_overlap_experiments.py` now follows the corrected v2 protocol. Use `analyze_routofk_overlap.py` only for old exploratory outputs.

## Example Codex Prompt

```text
Implement an exploratory r-out-of-K CIFAR-5 overlap study.

Use CIFAR-5 from CIFAR-10 classes 0,1,2,3,4. Train five heterogeneous EDL classifiers:
lenet, small_cnn, small_vgg, tiny_resnet, tiny_vit.

Create per-model training subsets with overlap ratios 1.0, 0.75, 0.5, 0.25, 0.0 and two partition modes:
stratified-balanced and class-skewed-private.

Export alpha, probability, belief, uncertainty, and plausibility for all five classifiers.
Evaluate only the r-out-of-K fusion rule for r=1..5.

Report accuracy, NLL, Brier, ECE, per-class selected r, and diversity metrics.
Do not fabricate data or metrics. All reported results must come from real model outputs.
```

## Caveat

This plan is superseded by Plan 02 because the original `best r` selection used the test split. Formal experiments must select `r` on `fusion_val` and report final metrics on held-out `test`.
