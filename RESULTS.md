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
| Macro precision | 0.6897 |
| Macro recall | 0.6754 |
| Macro F1 | 0.6820 |
| Cohen's kappa | 0.6214 |

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
