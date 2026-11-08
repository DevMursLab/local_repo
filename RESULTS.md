# Verified Experiment Results

This page is the authoritative record for results that may be cited from this repository. It intentionally separates measured evidence from planned work.

## Experiment E01: FER2013 categorical-emotion baseline

| Field | Value |
| --- | --- |
| Task | Seven-class facial-expression classification |
| Dataset | FER2013, obtained through its authorised Kaggle distribution |
| Split | Train 25,838 / validation 2,871 / held-out test 7,178 images |
| Model | ImageNet-initialised MobileNetV3-Large classifier |
| Selection rule | Best validation checkpoint; held-out test evaluated once |
| Hardware | NVIDIA Tesla T4 GPU (Kaggle) |
| Training | 25 epochs; class-balanced training |

### Held-out test results

| Metric | Score |
| --- | ---: |
| Accuracy | 0.6870 |
| 95% bootstrap accuracy CI (5,000 draws) | [0.6767, 0.6977] |
| Macro precision | 0.6897 |
| Macro recall | 0.6754 |
| Macro F1 | 0.6820 |
| Cohen's kappa | 0.6214 |

### Per-class analysis

| Class | F1 | Test support |
| --- | ---: | ---: |
| Angry | 0.6107 | 958 |
| Disgust | 0.7273 | 111 |
| Fear | 0.5509 | 1,024 |
| Happy | 0.8680 | 1,774 |
| Neutral | 0.6385 | 1,233 |
| Sad | 0.5588 | 1,247 |
| Surprise | 0.8199 | 831 |

The confidence interval was computed from 5,000 bootstrap resamples of the held-out predictions (seed 20261107). Fear and sad expressions remain the principal error modes; any future method should be evaluated against this class-wise baseline rather than accuracy alone.

## Experiment E03: fair backbone comparison

All models below use the same FER2013 split (25,838 train / 2,871 validation / 7,178 held-out test), seed 20261107, 25 epochs, class-balanced cross-entropy, ImageNet initialisation, and validation-selected checkpoints. This is a single-seed architecture comparison, not a multi-seed claim.

| Architecture | Test accuracy | Test macro F1 | Cohen's kappa |
| --- | ---: | ---: | ---: |
| MobileNetV3-Large | 0.6825 | 0.6791 | 0.6157 |
| EfficientNet-B0 | 0.6931 | 0.6891 | 0.6293 |
| ResNet-18 | **0.6992** | **0.6936** | **0.6363** |

Under this controlled comparison, ResNet-18 is the strongest tested architecture. The repeated-seed MobileNetV3 study remains useful as a lightweight-model robustness reference; it must not be directly compared to the single-seed ResNet-18 point estimate as though they had identical uncertainty analysis. A final manuscript should repeat the selected ResNet-18 configuration across the same three seeds before claiming an aggregate improvement.

## Experiment E04: selected-model robustness

The selected ResNet-18 configuration was repeated on three independently seeded, stratified train/validation splits. Each run used its own validation-selected checkpoint and the unchanged 7,178-image held-out test partition.

| Metric | Mean | Sample SD |
| --- | ---: | ---: |
| Accuracy | **0.6952** | 0.0037 |
| Macro precision | 0.6980 | 0.0049 |
| Macro recall | 0.6828 | 0.0045 |
| Macro F1 | **0.6891** | 0.0038 |
| Cohen's kappa | 0.6316 | 0.0044 |

The individual test accuracies were 0.6992, 0.6920, and 0.6945. This repeat establishes stable performance for the selected ResNet-18 configuration under the stated FER2013 protocol. It remains a benchmark-specific result, not a claim of state-of-the-art or cross-dataset generalisation.

Raw test predictions, the selected checkpoint, run configuration, and provenance JSON are retained as non-versioned research artifacts. The repository deliberately does not publish the dataset or model checkpoint because of size and data-distribution constraints.

## Experiment E02: validation-only architecture check

An EfficientNet-B0 comparison was run against the same manifest and evaluated on validation only. Its validation accuracy was **0.6907**, macro F1 **0.6597**, and kappa **0.6257** after 30 epochs. It did not improve macro F1 over the MobileNetV3 baseline, so it is not presented as the primary model. No held-out test score was used for this comparison.

## What these results do and do not establish

The completed evidence supports a reproducible **categorical FER baseline**. It does not establish:

- valence/arousal prediction quality (FER2013 does not provide continuous labels);
- music recommendation relevance or user satisfaction;
- clinical, therapeutic, usability, or human-study claims;
- state-of-the-art performance.

## Required path to a submission-ready study

1. Obtain an authorised dataset with continuous valence/arousal annotations and lock a subject-independent evaluation protocol.
2. Run dual-head perception experiments with confidence intervals and per-class/per-demographic error analysis where metadata permits.
3. Collect independent recommendation relevance labels or consented interaction data; compare against pre-registered baselines.
4. Conduct an ethics-approved human evaluation if making user-experience or wellbeing claims.
5. Package anonymised predictions, configuration, environment versions, and a statistical analysis plan with the manuscript.

See [the research execution protocol](paper/RESEARCH_EXECUTION_PROTOCOL.md) and [reproducibility guide](REPRODUCIBILITY.md) for operational details.
