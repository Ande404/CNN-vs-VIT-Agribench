# CNN-vs-VIT-Agribench
A systematic comparison of CNNs and Vision Transformers for plant disease classification and real-world robustness.

## Overview

CNN-vs-VIT-Agribench is a deep learning project that investigates the effectiveness of **Convolutional Neural Networks (CNNs)** and **Vision Transformers (ViTs)** for plant disease classification using leaf images. The project focuses on benchmarking these architectures under a controlled experimental setup and evaluating their ability to generalize to real-world data.

The primary goal is to understand the trade-offs between CNNs, which rely on local feature extraction, and Transformers, which model global relationships through self-attention.

---

## Objectives

* Build a plant disease classification system using leaf images
* Compare CNN and ViT architectures under identical conditions
* Evaluate model performance on both controlled and real-world datasets
* Analyze generalization, efficiency, and robustness

---

## Dataset

### Primary Dataset

* **PlantVillage Dataset**
* Subset of 5 classes:

  * Pepper Bell — Bacterial Spot
  * Pepper Bell — Healthy
  * Potato — Early Blight
  * Potato — Late Blight
  * Potato — Healthy

### Data Split

A custom stratified split was created to ensure fair comparison:

* Train: 80%
* Validation: 10%
* Test: 10%

All preprocessing and splitting steps are reproducible and stored in the `01_data_preparation.ipynb` notebook.

---

## Project Structure

```
agri-vision-bench/
│
├── data/
│   ├── raw/                # Original dataset
│   ├── subset/             # Selected classes
│   └── processed/          # Train/val/test split
│
├── notebooks/
│   ├── data_preparation.ipynb
│   ├── resnet50_baseline.ipynb
│   └── vit_baseline.ipynb (in progress)
│
├── model_checkpoints/
│   └── resnet50_best_checkpoint.pth
│
├── src/                    # (Optional future refactoring)
│   ├── datasets/
│   ├── models/
│   └── training/
│
├── requirements.txt
└── README.md
```

---

## Data Pipeline

* Image loading using PyTorch `ImageFolder`
* Preprocessing:
  * Resize to 224 × 224
  * Normalization using ImageNet statistics
* Data augmentation (training only):
  * Random horizontal flip
  * Random rotation
* Efficient batching with DataLoaders (GPU-enabled)

---

## CNN Baseline: ResNet-50

### Model

* Pretrained ResNet-50 (ImageNet)
* Final fully connected layer replaced for 5-class classification

### Training Setup

* Loss: CrossEntropyLoss
* Optimizer: Adam (learning rate = 1e-4)
* Batch size: 32
* Epochs: 5
* Hardware: GPU (T4)

### Checkpointing

* Best model saved based on validation accuracy
* Includes model state, optimizer state, and metadata

---

## Results (Preliminary)

| Metric              | Value      |
| ------------------- | ---------- |
| Train Accuracy      | ~99.7%     |
| Validation Accuracy | ~99.7–100% |

These results indicate strong performance on the controlled PlantVillage dataset.

---

## Next Steps

* Evaluate CNN baseline on test set
* Generate a confusion matrix and a classification report
* Implement Vision Transformer (ViT) baseline
* Perform comparative analysis:

  * Accuracy
  * F1-score
  * Training time
  * Inference speed
* Evaluate generalization on real-world dataset (PlantDoc)

---

## Reproducibility

* Fixed the random seed used for dataset splitting
* Shared dataset stored on Google Drive
* Consistent preprocessing and evaluation pipeline across models

---

## Notes

* The dataset is derived from a controlled environment, which may lead to inflated validation accuracy
* Additional evaluation on real-world datasets is required to assess generalization

---

## Contributors

* Andrew Luwaga
* Prince Kwarteng Amaning
* Varun Kadamanchi

---

## License

This project is for academic and research purposes.
