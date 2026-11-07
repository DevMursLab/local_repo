# AffectiveEcho: Multimodal Affective Music Recommendation

> Research prototype under evidence-building. It is not yet a submission-ready
> clinical or empirically validated system.

## What is implemented

- A dual-head facial-affect architecture for categorical emotion and continuous
  valence–arousal prediction.
- Face detection/alignment, Grad-CAM support, audio-feature extraction, and an
  Iso-Principle-inspired trajectory recommender with MMR diversification.
- A live demo and unit tests for engineering components.

## Evidence status

The historical file `experiments/benchmark_results.json` was made from a
simulation. Its values are not research findings and must not be reported in a
paper, poster, CV, or submission. The current manuscript/supplement therefore
remain drafts, not upload-ready submission files.

The publication path is in
[paper/RESEARCH_EXECUTION_PROTOCOL.md](paper/RESEARCH_EXECUTION_PROTOCOL.md).
It defines the required datasets, splits, checkpoint provenance, independent
evaluation, uncertainty, and human-study/ethics evidence.

## Real-result evaluation

After training a model on licensed data and exporting held-out predictions plus
paired session metrics, run:

```powershell
python experiments/evaluate_real_results.py `
  --classification artifacts/test_predictions.csv `
  --recommendations artifacts/session_metrics.csv `
  --proposed-method AffectiveEcho `
  --baseline-method GreedyCosine
```

The evaluator accepts only real CSV inputs and emits traceable metrics, a 95%
bootstrap confidence interval for accuracy, and a paired Wilcoxon result.

## Developer checks

```powershell
pytest -q
python experiments/evaluate_real_results.py --help
```

`experiments/benchmark_runner.py` is retained solely as a prototype simulation;
it is not part of the publication workflow.
