# Reproducibility

This repository deliberately excludes data, model checkpoints, raw participant
responses, and Kaggle run outputs. They may contain licensed material or are
too large for source control.

## Categorical FER baseline

1. Obtain FER2013 from its authorised Kaggle source and place it under
   `data/external/fer2013`.
2. Build and validate the manifest:

   ```powershell
   python experiments/prepare_fer2013.py
   python experiments/validate_manifest.py --manifest artifacts/fer2013_manifest.csv
   ```

3. Train locally or use the private `kaggle_kernel/` package with a GPU.
4. Keep test predictions and `provenance.json` outside Git. Use them to
   regenerate reported metrics rather than copying values by hand.

## Scientific boundaries

FER2013 supports categorical emotion classification only. It does not provide
continuous valence/arousal annotations and cannot justify clinical, therapeutic,
or human-subject claims. See [the full evidence protocol](paper/RESEARCH_EXECUTION_PROTOCOL.md).
