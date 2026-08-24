# Landcover Classification on DeepGlobe with Nested U-Net (UNet++)

A PyTorch Lightning implementation of multi-class land cover segmentation on the DeepGlobe Land Cover Classification dataset, using a Nested U-Net (UNet++) architecture with a pretrained RegNetY encoder.

## Overview

This project segments satellite images into multiple land cover categories (e.g. urban, agriculture, rangeland, forest, water, barren, unknown) using a UNet++ model from [`segmentation_models_pytorch`](https://github.com/qubvel/segmentation_models.pytorch). Training, validation, and testing are orchestrated with `pytorch_lightning`, and data augmentation is handled with `albumentations`.

## Dataset

[DeepGlobe Land Cover Classification Dataset](https://www.kaggle.com/datasets/balraj98/deepglobe-land-cover-classification-dataset) (Kaggle).

Each image has a corresponding RGB-encoded mask, where each color represents a distinct land cover class (defined in `class_dict.csv`). The notebook converts these RGB masks into single-channel category masks for training.

The dataset is split into:
- **Train**: ~60% (75% of the full set, then split again 80/20 for train/val)
- **Validation**: ~15%
- **Test**: 25%

## Model Architecture

- **Architecture**: UNet++ (Nested U-Net) via `segmentation_models_pytorch.UnetPlusPlus`
- **Encoder**: `timm-regnety_120`, pretrained on ImageNet
- **Input channels**: 3 (RGB)
- **Output classes**: number of land cover classes defined in `class_dict.csv`
- **Activation**: Softmax
- **Loss**: Multiclass Dice Loss

**Optimizer:** AdamW (`lr=1e-4`)
**Framework:** PyTorch Lightning (`pl.LightningModule` + `pl.LightningDataModule`)
**Callbacks:** Model checkpointing (best model by validation F1 score) and early stopping (on validation accuracy, patience = 5)
**Logging:** CSV logger, with training/validation/test curves for Loss, IoU, Accuracy, Precision, Recall, and F1 score

## Data Augmentation

Using `albumentations`:
- Resize to 320×320
- Random horizontal flip (p=0.5)
- Random vertical flip (p=0.5)
- Random brightness/contrast (p=0.3)

## Evaluation Metrics

Computed per epoch (train/val/test) via `segmentation_models_pytorch.metrics`:
- IoU score
- Accuracy
- Precision
- Recall
- F1 score

## Requirements

- Python 3.8+
- PyTorch
- pytorch-lightning
- segmentation-models-pytorch
- albumentations
- torchinfo
- opencv-python (`cv2`)
- numpy, pandas, scikit-learn, matplotlib
- tqdm

Install dependencies:

```bash
pip install torch torchvision
pip install git+https://github.com/PyTorchLightning/pytorch-lightning
pip install git+https://github.com/qubvel/segmentation_models.pytorch
pip install git+https://github.com/albumentations-team/albumentations
pip install torchinfo opencv-python pandas scikit-learn matplotlib tqdm
```

> A CUDA-capable GPU is strongly recommended — the notebook is configured to train with `accelerator="gpu"`. For CPU-only environments, change the `Trainer` accelerator setting accordingly.

## Usage

1. Clone this repository:
   ```bash
   git clone https://github.com/<your-username>/landcover-classification-unet.git
   cd landcover-classification-unet
   ```
2. Install the dependencies listed above.
3. Download the [DeepGlobe Land Cover Classification dataset](https://www.kaggle.com/datasets/balraj98/deepglobe-land-cover-classification-dataset) from Kaggle and place it so the notebook's expected paths resolve, e.g.:
   ```
   ../input/deepglobe-land-cover-classification-dataset/
   ├── class_dict.csv
   └── train/
       ├── *.jpg   (images)
       └── *.png   (masks)
   ```
   Adjust the paths in the notebook if you store the data elsewhere.
4. Open and run `landcover-classification-unet.ipynb` in Jupyter Notebook / JupyterLab / Kaggle / Google Colab.

The notebook will:
1. Load and split the dataset (train / val / test)
2. Convert RGB masks to category masks
3. Apply data augmentations
4. Build the UNet++ model with a pretrained RegNetY encoder
5. Train with early stopping and checkpointing (`Trainer.fit`)
6. Evaluate on the test set (`Trainer.test`)
7. Plot training/validation curves for loss and all metrics

## Project Structure

```
.
├── landcover-classification-unet.ipynb   # Main notebook (data, model, training, evaluation)
├── checkpoints/                          # Saved model checkpoints (generated during training)
├── lightning_logs/                       # CSV training/validation/test logs (generated during training)
└── README.md
```

## Results

After training, the notebook logs per-epoch metrics (Loss, IoU, Accuracy, Precision, Recall, F1 score) for train/val/test splits and plots them for visual comparison.

## License

Add a license of your choice (e.g. MIT) if you intend to make this repository public.

## Acknowledgements

- Dataset: [DeepGlobe Land Cover Classification Dataset](https://www.kaggle.com/datasets/balraj98/deepglobe-land-cover-classification-dataset) on Kaggle
- Model library: [segmentation_models.pytorch](https://github.com/qubvel/segmentation_models.pytorch)
- Architecture: [UNet++: A Nested U-Net Architecture for Medical Image Segmentation](https://arxiv.org/abs/1807.10165) (Zhou et al., 2018)
