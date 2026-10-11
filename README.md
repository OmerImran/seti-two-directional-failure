# Two Directions of Failure in a CNN Technosignature Detector

Code, injection manifest, and per-sample results for the paper:

> **Omer Imran (2026). *Two Directions of Failure: Missed Detections and False Alarms in a CNN Technosignature Detector on Breakthrough Listen Data.*** Preprint, not peer reviewed yet. (Link to be added when I upload on arXiv)

One frozen EfficientNet-B0 detector, trained once on the Kaggle *SETI Breakthrough Listen — E.T. Signal Search* data and never retrained. 15,000 synthetic signals injected with `setigen` into 300 real backgrounds, across two directions:

- **Direction 1 — misses:** five signal morphologies the detector was not trained on → recall.
- **Direction 2 — false alarms:** five RFI-like morphologies injected into ON *and* OFF panels → false-positive rate.

Plus Grad-CAM analysis of both failure modes and a training-free ON/OFF cadence filter.

## Headline numbers

All recomputable from the CSVs in this repo with `scripts/verify_results.py`.

| Quantity | Value |
|---|---|
| Baseline AUC (validation fold) | 0.8545 |
| Threshold (99th pct of validation negatives → 1% FPR) | 0.1309 |
| Baseline recall at threshold | 0.5525 [0.524, 0.580] |
| Clean-control FPR (300 uninjected backgrounds) | 4/300 = 0.013 [0.005, 0.034] |
| D1-1 extreme drift, amplitude 0.5 — recall | 0.410 |
| D1-3 S-shaped, amplitude 4.0 — recall | 0.900 |
| D2-5 high drift (in all six panels), amplitude 0.5 — FPR | 0.627 |

Brackets are 95% Wilson intervals; n = 300 per family × amplitude. See the paper for interpretation and limitations (single training run, single seed, synthetic morphologies, one filter design tested).

## Repository layout

```
notebooks/
  seti-baseline-v1.ipynb      training notebook (adapted from tuckerarrants, see Credits)
  seti-experiments-v1.ipynb   injection, scoring, Grad-CAM, filter
data/
  injection_manifest.csv      15,000 rows — THE dataset (see below)
  oof_df.csv                  out-of-fold validation predictions (AUC / threshold source)
results/                      per-sample and summary outputs (table below)
figures/                      figures as they appear in the paper
paper/                        main.tex, references, compiled PDF
scripts/verify_results.py     recomputes the headline numbers, no GPU needed
weights/                      see weights/README.md
```

### Result files

| File | Contents |
|---|---|
| `phase3_scores.csv` | Model score for every injected sample (15,000 rows; join to manifest on `sample_id`) |
| `phase3_summary.csv` | Recall/FPR per family × amplitude (50 rows) |
| `clean_control_scores.csv` | Scores for the 300 uninjected backgrounds |
| `phase4_cohort.csv` | Grad-CAM cohort: 6 groups × 12, selection rule fixed before viewing CAMs (seed 4141) |
| `phase4_ratios.csv` | Attention measures per cohort sample |
| `phase4_deletion.csv` | Deletion curves, CAM-ranked vs 5 seeded random orders |
| `phase4_occlusion.csv` | Occlusion test per cohort sample |
| `phase5_energy_table.csv` | Injected energy vs model response per family |
| `phase55_veto.csv` | Cadence-filter veto rates per family × amplitude |
| `phase55_per_sample_S.csv` | Filter statistic S for every sample |
| [ADD REMAINING FILES] | |

## The injection manifest is the dataset

No injected spectrograms are stored. Injection is deterministic: every sample is regenerated from its manifest row (`sample_id, family, direction, label, amplitude, freq_pos, seed, source_id, panels`) plus the source background from the competition data.

**Do not regenerate the manifest.** The original builder derived seeds with Python's built-in `hash()`, which is randomised per process. Re-running it produces different seeds and positions. The notebook loads `injection_manifest.csv` instead; always use the saved file.

## Reproducing

**Quick check (any machine, ~seconds):**
```bash
pip install -r requirements.txt
python scripts/verify_results.py
```

**Full re-run (Kaggle):**
1. Accept the competition rules and attach *SETI Breakthrough Listen — E.T. Signal Search*. Use `train/` only; ignore `old_leaky_data/`.
2. Attach the frozen weights and `pytorch-image-models` (offline `timm`).
3. Accelerator **T4** — current PyTorch has dropped P100 (`sm_60`) support.
4. Session persistence: **Files only**.
5. Run `seti-experiments-v1.ipynb`. Scoring all 15,000 samples takes ~6.5 min on a T4.

Things that silently break results if wrong:
- Checkpoint must be `efficientnet_b0_fold0_best_score.pth`, **not** `best_loss`.
- Input recipe: ON panels only (`image[::2]`) → `float32` → stack and transpose to `(256, 819)` → resize to 512×512 → tensor. **No normalisation.**
- The model outputs a logit; apply `sigmoid` before comparing to 0.1309.
- Data are `float16`; cast to `float32` before any statistics (`np.percentile` returns NaN on float16).
- Test-time augmentation is off throughout so Grad-CAM matches the scored prediction.

## Data

The Breakthrough Listen spectrograms are **not redistributed** here; they are subject to the Kaggle competition's rules. Everything in this repo is either my own output or derived from it.

## Credits

- Baseline training notebook adapted from **tuckerarrants, "SETI Learned Image Resizing"** (Kaggle). Architecture and training loop are theirs; I trained it once and froze it.
- Signal injection: `setigen` (Brzycki et al.).
- Grad-CAM: Selvaraju et al. 2017. EfficientNet: Tan & Le 2019.

## AI-assistance disclosure

AI assistance. I used Claude (Anthropic) as an assistant during this project: to help debug code, revise a few portions of the text, and review the manuscript. The research question, experimental design, pre-specified decision rules, and all analysis decisions are my own. I ran all experiments myself on Kaggle, and every number reported comes from those saved outputs.

## License

Code: MIT (see `LICENSE`). Manifest, results, and figures: CC BY 4.0.

## Citation

See `CITATION.cff`, or use the Zenodo DOI: [gotta add ts too]
