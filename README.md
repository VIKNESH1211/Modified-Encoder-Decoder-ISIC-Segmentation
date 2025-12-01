## Modified Encoder - Decoder

### Results (from Segmentation_Binary.ipynb)

Best validation metrics observed over 20 epochs for each encoder variant.

| Encoder         | Best Val Dice | Val IoU | Best Val Accuracy | Best Val Loss | Notes                                                                |
| --------------- | ------------: | ----------------: | ----------------: | ------------: | -------------------------------------------------------------------- |
| VGG16           |        0.9299 |            0.8692 |            96.82% |        0.0839 | Taken from the highest recorded validation Dice in the VGG16 run.    |
| ResNet-18       |        0.9318 |            0.8721 |            96.96% |        0.0800 | Highest validation Dice observed across epochs in the ResNet-18 run. |
| ResNeXt50 32x4d |        0.9297 |            0.8690 |            96.88% |        0.0924 | Highest validation Dice observed in the ResNeXt50 run.               |
| InceptionV4     |        0.9272 |            0.8647 |            96.81% |        0.0837 | From epoch with best observed validation Dice in notebook output.    |
| EfficientNet-B7 |        0.9402 |            0.8872 |            97.44% |        0.0719 | From epoch with best observed validation Dice in notebook output.    |

> If additional epochs surpass these metrics, update the table accordingly.

#### Test Results

Reported performance on the held-out test set.

| Encoder         | Test Loss | Test Accuracy | Test Dice | Test IoU |
| --------------- | --------: | ------------: | --------: | -----------------: |
| EfficientNet-B7 |    0.0751 |        97.20% |    0.9422 |             0.8906 |
| InceptionV4     |    0.0784 |        97.04% |    0.9393 |             0.8856 |
| ResNeXt50 32x4d |    0.0808 |        96.89% |    0.9363 |             0.8799 |
| ResNet-18       |    0.0874 |        96.73% |    0.9332 |             0.8747 |
| VGG16           |    0.0897 |        96.47% |    0.9308 |             0.8705 |

---

Reusable, CLI-driven training entry points for binary image segmentation using U-Net with different encoders from `segmentation_models_pytorch`. The code generalizes the logic from the `Segmentation_Binary.ipynb` notebook into a structured Python package.

### Folder structure
```
Modified Encoder - Decoder/
├─ __init__.py
├─ common/
│  ├─ __init__.py
│  ├─ data.py
│  ├─ train_utils.py
│  └─ runner.py
├─ unet_vgg16/
│  ├─ __init__.py
│  └─ unet_vgg16_training.py
├─ unet_resnet18/
│  ├─ __init__.py
│  └─ unet_resnet18_training.py
├─ unet_resnext50_32x4d/
│  ├─ __init__.py
│  └─ unet_resnext50_32x4d_training.py
├─ unet_inceptionv4/
│  ├─ __init__.py
│  └─ unet_inceptionv4_training.py
└─ unet_efficientnet_b7/
   ├─ __init__.py
   └─ unet_efficientnet_b7_training.py
```

### Requirements
- Python 3.9+
- PyTorch (with CUDA if available)
- segmentation-models-pytorch
- albumentations, albumentations[imgaug] (optional), opencv-python
- scikit-learn, tqdm

Example install (PowerShell):
```powershell
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install segmentation-models-pytorch albumentations opencv-python scikit-learn tqdm
```
Adjust CUDA/PyTorch versions as needed.

### Dataset format
- Images live in a directory (e.g., `.../ISBI2016_ISIC_Part1_Training_Data`)
- Masks live in a parallel directory (e.g., `.../ISBI2016_ISIC_Part1_Training_GroundTruth`)
- Default mask filename pattern is `{stem}_Segmentation.png` for each `image.jpg`
  - You can change this via `--mask-template` and supported `--image-extensions`.

### Running training
Every encoder folder provides a training script that exposes a uniform CLI. Paths below are shown for Windows PowerShell; quote paths with spaces.

Minimal example (ResNet-18 encoder):
```powershell
python ".\Modified Encoder - Decoder\unet_resnet18\unet_resnet18_training.py" ^
  --train-image-dir "D:\data\ISIC\ISBI2016_ISIC_Part1_Training_Data" ^
  --train-mask-dir  "D:\data\ISIC\ISBI2016_ISIC_Part1_Training_GroundTruth"
```

With validation directory explicitly provided:
```powershell
python ".\Modified Encoder - Decoder\unet_resnet18\unet_resnet18_training.py" ^
  --train-image-dir "D:\data\ISIC\train_images" ^
  --train-mask-dir  "D:\data\ISIC\train_masks" ^
  --val-image-dir   "D:\data\ISIC\val_images" ^
  --val-mask-dir    "D:\data\ISIC\val_masks"
```

Train and then evaluate on a held-out test set:
```powershell
python ".\Modified Encoder - Decoder\unet_vgg16\unet_vgg16_training.py" ^
  --train-image-dir "D:\data\ISIC\ISBI2016_ISIC_Part1_Training_Data" ^
  --train-mask-dir  "D:\data\ISIC\ISBI2016_ISIC_Part1_Training_GroundTruth" ^
  --epochs 20 --batch-size 8 --image-size 256 --optimizer adam --lr 1e-4 ^
  --output-path ".\outputs\unet_vgg16_weights.pth" ^
  --test-image-dir "D:\data\ISIC\ISBI2016_ISIC_Part1_Test_Data" ^
  --test-mask-dir  "D:\data\ISIC\ISBI2016_ISIC_Part1_Test_GroundTruth"
```

Switching encoders is as simple as calling a different script:
- VGG16: `unet_vgg16\unet_vgg16_training.py`
- ResNet-18: `unet_resnet18\unet_resnet18_training.py`
- ResNeXt50 32x4d: `unet_resnext50_32x4d\unet_resnext50_32x4d_training.py`
- InceptionV4: `unet_inceptionv4\unet_inceptionv4_training.py`
- EfficientNet-B7: `unet_efficientnet_b7\unet_efficientnet_b7_training.py`

### Key CLI arguments
Data:
- `--train-image-dir` (required): training images directory
- `--train-mask-dir` (required): training masks directory
- `--val-image-dir`, `--val-mask-dir`: optional explicit validation split
- `--mask-template`: mask filename pattern (default `{stem}_Segmentation.png`)
- `--image-extensions`: extensions to include (default: jpg jpeg png)

Training:
- `--epochs` (default 20)
- `--batch-size` (default 8)
- `--num-workers` (default 4)
- `--image-size` (default 256)
- `--val-split` (default 0.2 if val dirs not provided)
- `--augment` (default true)

Optimization and loss:
- `--optimizer` {adam, adamw, sgd} (default adam)
- `--lr` (default 1e-4)
- `--weight-decay` (default 0.0)
- `--loss` {dice, focal, bce, tversky} (default dice)

Model:
- `--encoder-weights` (default imagenet; use `none` for random init)
- `--in-channels` (default 3)
- `--classes` (default 1 for binary)

Runtime and outputs:
- `--threshold` (default 0.5) for binarizing predictions when computing metrics
- `--monitor-metric` {dice, accuracy, loss} (default dice) for checkpointing
- `--device` {auto, cuda, cpu} (default auto)
- `--seed` (default 42)
- `--output-path` path to save best weights (defaults per-encoder)
- `--history-path` optional JSON file to save per-epoch train/val metrics

### Outputs
- Best model weights saved to `--output-path` when validation improves on the selected `--monitor-metric`.
- Optional training history JSON if `--history-path` is provided.
- If `--test-image-dir` and `--test-mask-dir` are provided, a final evaluation is printed after training.

### Tips
- Ensure masks are binary (0/255). The loader internally binarizes to {0,1}.
- If your filenames differ, adapt `--mask-template` (e.g., `{stem}_mask.png`).
- Use `--encoder-weights none` to train from scratch.

### License
This repository is for research and educational purposes. Refer to the original datasets’ licenses for usage restrictions.


