# Modified Encoder–Decoder Architectures for Binary Image Segmentation

This repository presents a modular and extensible framework for **binary image segmentation** using **U-Net** architectures with a variety of pretrained encoder backbones. It enables reproducible experimentation, comparison, and benchmarking of multiple encoder variants under a unified training pipeline.

---

## 🧠 Overview

Each encoder variant is implemented as a separate submodule with a consistent command-line interface (CLI). The framework leverages **`segmentation_models_pytorch`** for model definitions and supports a range of encoders including **VGG16**, **ResNet-18**, **ResNeXt50 (32×4d)**, **InceptionV4**, and **EfficientNet-B7**.

---

## 📊 Experimental Results

### Validation Performance (Best Epoch over 20 Epochs)

| Encoder               | Best Val Dice | Best Val Accuracy | Best Val Loss | Notes                                                       |
| --------------------- | ------------: | ----------------: | ------------: | ----------------------------------------------------------- |
| **VGG16**             |        0.9299 |            96.82% |        0.0839 | Highest recorded validation Dice during VGG16 run.          |
| **ResNet-18**         |        0.9318 |            96.96% |        0.0800 | Peak validation Dice observed in ResNet-18 experiment.      |
| **ResNeXt50 (32×4d)** |        0.9297 |            96.88% |        0.0924 | Maximum validation Dice achieved in ResNeXt50 run.          |
| **InceptionV4**       |        0.9272 |            96.81% |        0.0837 | Best validation Dice epoch recorded in InceptionV4 run.     |
| **EfficientNet-B7**   |    **0.9402** |        **97.44%** |    **0.0719** | Superior validation Dice across all encoder configurations. |

*Note: Update this table if subsequent epochs yield improved metrics.*

### Test Set Evaluation

| Encoder               |  Test Loss | Test Accuracy |  Test Dice |
| --------------------- | ---------: | ------------: | ---------: |
| **EfficientNet-B7**   | **0.0751** |    **97.20%** | **0.9422** |
| **InceptionV4**       |     0.0784 |        97.04% |     0.9393 |
| **ResNeXt50 (32×4d)** |     0.0808 |        96.89% |     0.9363 |
| **ResNet-18**         |     0.0874 |        96.73% |     0.9332 |
| **VGG16**             |     0.0897 |        96.47% |     0.9308 |

These results demonstrate consistent segmentation performance across architectures, with **EfficientNet-B7** achieving the best overall accuracy and Dice coefficient.

---

## 📁 Repository Structure

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

---

## ⚙️ Requirements

* Python ≥ 3.9
* PyTorch (with CUDA if available)
* `segmentation-models-pytorch`
* `albumentations`, `opencv-python`
* `scikit-learn`, `tqdm`

**Example installation (PowerShell):**

```powershell
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install segmentation-models-pytorch albumentations opencv-python scikit-learn tqdm
```

Adjust CUDA and PyTorch versions according to your hardware setup.

---

## 🧩 Dataset Organization

The dataset should follow a parallel directory structure:

```
📂 ISIC/
├─ ISBI2016_ISIC_Part1_Training_Data/
│  ├─ image_1.jpg
│  ├─ image_2.jpg
│  └─ ...
└─ ISBI2016_ISIC_Part1_Training_GroundTruth/
   ├─ image_1_Segmentation.png
   ├─ image_2_Segmentation.png
   └─ ...
```

* Mask filenames follow the pattern `{stem}_Segmentation.png`.
* The pattern can be modified using `--mask-template` if necessary.
* Supported image extensions include `jpg`, `jpeg`, and `png`.

---

## 🏃‍♂️ Training and Evaluation

Each encoder directory includes an independent training script that exposes a consistent CLI.

### Example 1 — Minimal Training (ResNet-18)

```powershell
python ".\Modified Encoder - Decoder\unet_resnet18\unet_resnet18_training.py" ^
  --train-image-dir "D:\data\ISIC\ISBI2016_ISIC_Part1_Training_Data" ^
  --train-mask-dir  "D:\data\ISIC\ISBI2016_ISIC_Part1_Training_GroundTruth"
```

### Example 2 — Explicit Validation Split

```powershell
python ".\Modified Encoder - Decoder\unet_resnet18\unet_resnet18_training.py" ^
  --train-image-dir "D:\data\ISIC\train_images" ^
  --train-mask-dir  "D:\data\ISIC\train_masks" ^
  --val-image-dir   "D:\data\ISIC\val_images" ^
  --val-mask-dir    "D:\data\ISIC\val_masks"
```

### Example 3 — Train and Evaluate (VGG16 Encoder)

```powershell
python ".\Modified Encoder - Decoder\unet_vgg16\unet_vgg16_training.py" ^
  --train-image-dir "D:\data\ISIC\ISBI2016_ISIC_Part1_Training_Data" ^
  --train-mask-dir  "D:\data\ISIC\ISBI2016_ISIC_Part1_Training_GroundTruth" ^
  --epochs 20 --batch-size 8 --image-size 256 ^
  --optimizer adam --lr 1e-4 ^
  --output-path ".\outputs\unet_vgg16_weights.pth" ^
  --test-image-dir "D:\data\ISIC\ISBI2016_ISIC_Part1_Test_Data" ^
  --test-mask-dir  "D:\data\ISIC\ISBI2016_ISIC_Part1_Test_GroundTruth"
```

### Switching Encoder Backbones

* **VGG16:** `unet_vgg16\unet_vgg16_training.py`
* **ResNet-18:** `unet_resnet18\unet_resnet18_training.py`
* **ResNeXt50 (32×4d):** `unet_resnext50_32x4d\unet_resnext50_32x4d_training.py`
* **InceptionV4:** `unet_inceptionv4\unet_inceptionv4_training.py`
* **EfficientNet-B7:** `unet_efficientnet_b7\unet_efficientnet_b7_training.py`

---

## 🧮 Key CLI Parameters

### Data Configuration

| Argument                            | Description                                           |
| ----------------------------------- | ----------------------------------------------------- |
| `--train-image-dir`                 | Path to training images *(required)*                  |
| `--train-mask-dir`                  | Path to training masks *(required)*                   |
| `--val-image-dir`, `--val-mask-dir` | Optional validation directories                       |
| `--mask-template`                   | Filename pattern (default: `{stem}_Segmentation.png`) |
| `--image-extensions`                | Image extensions to include                           |

### Training Parameters

| Argument        | Description                                  |
| --------------- | -------------------------------------------- |
| `--epochs`      | Default: 20                                  |
| `--batch-size`  | Default: 8                                   |
| `--num-workers` | Default: 4                                   |
| `--image-size`  | Default: 256                                 |
| `--val-split`   | Default: 0.2 (if no validation set provided) |
| `--augment`     | Apply data augmentation (default: True)      |

### Optimization

| Argument         | Description                                     |
| ---------------- | ----------------------------------------------- |
| `--optimizer`    | `{adam, adamw, sgd}` (default: `adam`)          |
| `--lr`           | Learning rate (default: `1e-4`)                 |
| `--weight-decay` | Default: `0.0`                                  |
| `--loss`         | `{dice, focal, bce, tversky}` (default: `dice`) |

### Model and Runtime

| Argument            | Description                                |
| ------------------- | ------------------------------------------ |
| `--encoder-weights` | Pretrained weights (default: `imagenet`)   |
| `--in-channels`     | Input channels (default: 3)                |
| `--classes`         | Output classes (default: 1 for binary)     |
| `--threshold`       | Threshold for binarization (default: 0.5)  |
| `--monitor-metric`  | `{dice, accuracy, loss}` (default: `dice`) |
| `--device`          | `{auto, cuda, cpu}` (default: auto)        |
| `--seed`            | Random seed (default: 42)                  |
| `--output-path`     | Save path for best weights                 |
| `--history-path`    | Optional JSON file to store epoch metrics  |

---

## 📈 Outputs

* Best model weights saved automatically based on the monitored metric (`--monitor-metric`).
* Optional training history (`.json`) file containing epoch-wise metrics.
* Automatic test evaluation if test directories are provided.

---

## 🧩 Practical Notes

* Ensure ground truth masks are **binary (0/255)**; the loader automatically converts them to `{0,1}`.
* Adapt `--mask-template` for custom naming schemes (e.g., `{stem}_mask.png`).
* Use `--encoder-weights none` to train models from scratch.

---

## 📜 License

This repository is intended for **research and educational purposes**. Please refer to the respective dataset licenses for any restrictions regarding redistribution or commercial use.

--
