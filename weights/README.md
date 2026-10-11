# Frozen weights

`efficientnet_b0_fold0_best_score.pth` (16.44 MB) — EfficientNet-B0 with learned resizer, trained once
(15 epochs, seed 42, fold 0, mode `spatial_3`), never retrained. Every result in the paper uses it.

Download: [LINK — GitHub release asset or Zenodo]

Load with `torch.load(path, map_location="cpu", weights_only=False)` — the file also stores a
NumPy array of predictions, which PyTorch ≥ 2.6 refuses under the default `weights_only=True`.

Use `best_score`, never `best_loss`.
