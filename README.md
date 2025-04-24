# Knife Classification with Deep Learning Models

![Deep Learning](https://img.shields.io/badge/Deep%20Learning-CNN-blue)
![Python](https://img.shields.io/badge/Python-3.x-green)
![PyTorch](https://img.shields.io/badge/PyTorch-1.x-red)

This repository implements and compares various deep learning models for knife classification, including custom CNNs, EfficientNet variants, ResNet architectures, and ensemble methods.

## Table of Contents
- [Key Features](#key-features)
- [Implemented Models](#implemented-models)
- [Optimal Configurations](#optimal-configurations)
- [Installation](#installation)
- [Usage](#usage)
- [Dataset Information](#dataset-information)
- [Results](#results)
- [References](#references)
- [License](#license)

## Key Features

✔ Comprehensive model comparison (transfer vs non-transfer learning)  
✔ Hyperparameter optimization (learning rate, batch size, epochs)  
✔ Advanced techniques like Cosine Annealing LR Scheduler  
✔ Ensemble modeling combining best architectures  
✔ Evaluation using mAP, Top 1, and Top 5 accuracy metrics  

## Implemented Models

1. **Custom CNN** (trained from scratch)
2. **EfficientNet Series** (B0-B7)
3. **EfficientNetV2 Models** (S, M, L)
4. **ResNet Models** (18, 34, 50, 101, 152)
5. **Ensemble Model** (EfficientNetV2-L + ResNet-152)

## Optimal Configurations

| Model               | Learning Rate | Batch Size | Epochs | Best mAP  |
|---------------------|---------------|------------|--------|-----------|
| Ensemble Model      | 0.00005       | 8          | 9      | **0.8349**|
| EfficientNetV2-L    | 0.0001        | 8          | 6      | 0.7909    |
| ResNet-152          | 0.00005       | 8          | 9      | 0.8009    |
| EfficientNet-B4     | 0.0001        | 8          | 8      | 0.7800    |
| Custom CNN          | 0.0001        | 8          | 20     | 0.1576    |

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/knife-classification.git
cd knife-classification

2. Install dependencies:

```bash
pip install -r requirements.txt

3. Usage
Training
```bash
python train.py \
  --model [MODEL_NAME] \
  --lr [LEARNING_RATE] \
  --batch_size [BATCH_SIZE] \
  --epochs [NUM_EPOCHS]

Example for best-performing ensemble model:

```bash
python train.py --model ensemble --lr 0.00005 --batch_size 8 --epochs 9
Testing
```bash
python test.py \
  --model [MODEL_NAME] \
  --weights_path [PATH_TO_WEIGHTS]
## Dataset Information
192 knife classes

Image size: 224×224 pixels

CSV files included for data structure reference

Actual dataset available upon request

Results
Key findings:

Ensemble model achieved highest mAP (0.8349)

Transfer learning models outperformed custom CNN by >400%

Optimal learning rates: 0.00005-0.0001

Best batch sizes: 8-16

Cosine Annealing LR Scheduler improved convergence
