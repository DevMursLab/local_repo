# Submission Checklist — FER2013 Baseline Paper

The current draft is intentionally scoped to categorical facial-expression recognition. Complete every item before submitting it to a venue.

## Ready now

- [x] Held-out test result is recorded with a fixed split and seed.
- [x] The primary model was selected using validation, not test accuracy.
- [x] The paper states its scope and limitations.
- [x] Training, validation, and result-evaluation scripts are versioned.
- [x] Repository authoring history names only Mursalin.

## Must complete before submission

- [ ] Select a specific venue and replace the generic IEEE template with its official template.
- [ ] Confirm the venue permits a single-baseline reproducibility paper; otherwise add pre-registered baselines.
- [ ] Run at least three additional fixed seeds and report mean, standard deviation, and confidence intervals.
- [ ] Export and report a per-class precision/recall/F1 table and confusion matrix.
- [ ] Freeze a release tag; archive the exact source, environment lockfile, configuration, result JSON, and prediction checksum.
- [ ] Verify FER2013 redistribution/citation terms and cite the dataset source exactly as required.
- [ ] Have a coauthor or independent reader check reproducibility and language.
- [ ] Run plagiarism, reference, grammar, and template compliance checks.

## Do not add without new evidence

- Valence/arousal performance claims.
- Music recommendation relevance, satisfaction, or playlist-quality claims.
- Clinical, therapeutic, wellbeing, or usability claims.
- ``state of the art'' language without a fair, documented comparison.
