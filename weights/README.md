# Frozen weights

`efficientnet_b0_fold0_best_score.pth` (16.44 MB): EfficientNet-B0 with a learned resizer, trained once
(15 epochs, seed 42, fold 0, mode `spatial_3`) and never retrained. Every result in the paper uses these weights.

Included in this folder.

Load with `torch.load(path, map_location="cpu", weights_only=False)`. The file also stores a
NumPy array of predictions, which PyTorch 2.6 and later refuse to load under the default `weights_only=True`.

Use `best_score`, never `best_loss`.