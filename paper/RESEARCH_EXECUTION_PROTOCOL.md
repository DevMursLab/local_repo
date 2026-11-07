# AffectiveEcho: publication-evidence protocol

`benchmark_runner.py` is a prototype simulation. Its output must never be cited,
plotted, or described as an empirical result. The existing manuscript and
supplement contain unverified performance and human-study statements and are
therefore **not submission-ready**.

## Evidence required before submission

1. Record each dataset's source, licence, version, exclusions, and split.
2. Train on train data only; select hyperparameters on validation data; run the
   test split once after locking the model.
3. Store checkpoint SHA-256, commit hash, package versions, seed, hardware,
   command, and raw per-sample predictions.
4. Evaluate recommendation quality with held-out interactions or a preregistered
   user study. Never derive relevance from the method's own target rule.
5. Use `experiments/evaluate_real_results.py` for reportable metrics, a 95% CI,
   and a paired Wilcoxon test. Retain both input CSV files.
6. Make clinical, therapeutic, privacy, fairness, and human-study claims only
   with suitable protocol, consent/ethics approval, and analysis.

## Required schemas

Classification CSV: `sample_id,y_true,y_pred` (one held-out sample per row).

Recommendation CSV: `session_id,method,ndcg_at_5` (one session-method pair per
row; methods must use identical sessions).

Train only from a licensed image manifest with
`image_path,split,label,valence,arousal`, where split is `train`, `validation`,
or `test`. The trainer selects its checkpoint using validation accuracy and
exports test predictions once:

```powershell
python experiments/train_dual_head.py --manifest artifacts/face_manifest.csv --output-dir artifacts/run_001
```

Validate the manifest first (including subject-disjoint splits when `subject_id`
is supplied):

```powershell
python experiments/validate_manifest.py --manifest artifacts/face_manifest.csv --output artifacts/manifest_audit.json
```

FER2013 is available in this workspace under `data/external/fer2013` and may be
prepared for a categorical-only baseline. It must not be used to claim
valence/arousal performance:

```powershell
python experiments/prepare_fer2013.py
python experiments/train_dual_head.py --manifest artifacts/fer2013_manifest.csv --output-dir artifacts/fer2013_run_001 --classification-only --class-balanced
```

```powershell
python experiments/evaluate_real_results.py `
  --classification artifacts/test_predictions.csv `
  --recommendations artifacts/session_metrics.csv `
  --proposed-method AffectiveEcho `
  --baseline-method GreedyCosine
```
