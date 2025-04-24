# 🔪 Knife Classification with Deep Learning Models

![Deep Learning](https://img.shields.io/badge/Deep%20Learning-CNN-blue)  
![Python](https://img.shields.io/badge/Python-3.x-green)  
![PyTorch](https://img.shields.io/badge/PyTorch-1.x-red)

> A deep dive into the world of sharp objects and sharper models — comparing CNNs, EfficientNets, ResNets, and ensemble techniques for knife image classification.

---

## 📚 Table of Contents
- [✨ Key Features](#-key-features)
- [🧠 Implemented Models](#-implemented-models)
- [⚙️ Optimal Configurations](#-optimal-configurations)
- [🚀 Installation](#-installation)
- [🛠️ Usage](#-usage)
- [🗂️ Dataset Information](#️-dataset-information)
- [📊 Results](#-results)
- [📖 References](#-references)
- [📝 License](#-license)

---

## ✨ Key Features

- ✔️ Comprehensive model comparison (Transfer vs Non-transfer Learning)  
- ✔️ Hyperparameter optimization: learning rate, batch size, epochs  
- ✔️ Cosine Annealing LR Scheduler for smoother convergence  
- ✔️ Ensemble learning to stack performance  
- ✔️ Evaluation using mAP, Top-1, and Top-5 Accuracy metrics  

---

## 🧠 Implemented Models

1. **Custom CNN** — built from scratch  
2. **EfficientNet Series** — from B0 to B7  
3. **EfficientNetV2** — S, M, L variants  
4. **ResNet** — 18, 34, 50, 101, 152  
5. **Ensemble Model** — EfficientNetV2-L + ResNet-152  

---

## ⚙️ Optimal Configurations

| Model               | Learning Rate | Batch Size | Epochs | Best mAP   |
|---------------------|---------------|------------|--------|------------|
| **Ensemble Model**  | 0.00005       | 8          | 9      | **0.8349** |
| EfficientNetV2-L    | 0.0001        | 8          | 6      | 0.7909     |
| ResNet-152          | 0.00005       | 8          | 9      | 0.8009     |
| EfficientNet-B4     | 0.0001        | 8          | 8      | 0.7800     |
| Custom CNN          | 0.0001        | 8          | 20     | 0.1576     |

---

## 🚀 Installation

1. Clone the repo:
   \`\`\`bash
   git clone https://github.com/yourusername/knife-classification.git
   cd knife-classification
   \`\`\`

2. Install dependencies:
   \`\`\`bash
   pip install -r requirements.txt
   \`\`\`

---

## 🛠️ Usage

### 🔧 Training

\`\`\`bash
python train.py \
  --model [MODEL_NAME] \
  --lr [LEARNING_RATE] \
  --batch_size [BATCH_SIZE] \
  --epochs [NUM_EPOCHS]
\`\`\`

**Example (Best Ensemble Model):**
\`\`\`bash
python train.py --model ensemble --lr 0.00005 --batch_size 8 --epochs 9
\`\`\`

### 🧪 Testing

\`\`\`bash
python test.py \
  --model [MODEL_NAME] \
  --weights_path [PATH_TO_WEIGHTS]
\`\`\`

---

## 🗂️ Dataset Information

- 📦 192 knife classes  
- 📏 Image Size: 224 × 224 pixels  
- 📑 CSV files included for reference structure  
- 📥 Full dataset available upon request  

---

## 📊 Results

- 🏆 **Ensemble model** crushed it with **0.8349 mAP**
- 🔄 Transfer learning > Custom CNN (by over 400% 👀)
- 🔍 Optimal learning rates: **0.00005–0.0001**
- 📦 Best batch sizes: **8–16**
- ⏳ Cosine Annealing Scheduler improved convergence magic  

---

## 📖 References

> Add relevant papers, repositories, or articles here if needed.

---

## 📝 License

> Add your license info here (e.g., MIT, Apache 2.0)
