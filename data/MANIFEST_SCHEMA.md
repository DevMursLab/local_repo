# Face-data manifest schema

Use one row per image. Do not commit copyrighted images, participant identities,
or sensitive raw metadata to this repository.

| Column | Required | Meaning |
| --- | --- | --- |
| `image_path` | yes | Local path to a licensed image; unique across all splits. |
| `split` | yes | Exactly `train`, `validation`, or `test`. |
| `label` | yes | One of: Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral. |
| `valence` | yes | Continuous annotation in [-1, 1]. |
| `arousal` | yes | Continuous annotation in [-1, 1]. |
| `subject_id` | strongly recommended | Stable anonymous participant/source ID. A subject must occur in one split only. |

Before training, run:

```powershell
python experiments/validate_manifest.py --manifest artifacts/face_manifest.csv --output artifacts/manifest_audit.json
```

The validator rejects duplicate images, invalid labels/ranges, missing splits,
and subject-level split leakage.
