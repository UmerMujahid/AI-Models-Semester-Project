# 🤖 AI Models — Semester Project
---

## 👥 Team Members

| Name | Roll No |
|------|---------|
| Umer Karamat | 23L-0873 |
| M Basim Irfan | 23L-0846 |
| Umer Mujahid | 23L-0774 |
| Muhammad Rehan | 23L-0925 |

---

## 📦 Repository Structure

```
ai-models/
│
├── RNN_Model/
│   ├── rnn.ipynb                # Main training notebook Bi-LSTM
│   └── DATA                     # Complete Dataset
│
├── CNN_Model/
│   ├── CNN_MODEL.ipynb         # CNN 2D from scratch notebook
│   ├── CNN_MODEL_2.ipynb       # MobileNetV2 ( Transfer Learning )
│   └── DATA                    # Complete Dataset
│
├── reports/
│   ├── LSTM_PM25_Project_Report.docx
│   └── CNN_Model_Project_Report.docx
│
└── README.md                     # ← You are here
```

---

## 📁 Project 1 — LSTM-Based PM2.5 Air Pollution Prediction

### 🎯 Objective
Predict hourly PM2.5 fine particulate matter concentrations using an LSTM neural network, and verify the core findings of *Li et al. (2017)* on a single-station dataset.

### 📊 Dataset
- **Source:** [Beijing PM2.5 Dataset — UCI Machine Learning Repository (via Kaggle)](https://www.kaggle.com/datasets/djhavera/beijing-pm25-data-data-set)
- **Period:** January 2010 – December 2014
- **Records:** ~43,000 hourly readings (after cleaning)
- **Station:** US Embassy, Beijing (single station)

| Feature | Role |
|---------|------|
| PM2.5 (µg/m³) | Target variable |
| Dew Point (°C) | Humidity proxy |
| Temperature (°C) | Atmospheric condition |
| Pressure (hPa) | Meteorological factor |
| Cumulative Wind Speed (m/s) | Pollutant dispersion proxy |
| Cumulative Hours of Snow | Weather event indicator |
| Cumulative Hours of Rain | Wet deposition indicator |

### 🏗️ Model Architecture

| Layer | Type | Configuration |
|-------|------|---------------|
| 1 | LSTM | 50 units, `return_sequences=False` |
| 2 | Dropout | Rate = 0.2 |
| 3 | Dense | 1 unit, linear activation |

- **Input shape:** `(24, 7)` — 24-hour sliding window × 7 features  
- **Optimizer:** Adam | **Loss:** Mean Squared Error (MSE)  
- **Epochs:** 20 | **Batch size:** 72

### 📈 Results

| Model | RMSE (µg/m³) | Notes |
|-------|-------------|-------|
| **Our LSTM (Single-Station)** | **24.55** | UCI dataset, lightweight |
| LSTME — Li et al. (2017) | 12.60 | 12 stations, spatiotemporal |
| LSTM NN — Li et al. (2017) | 17.94 | Multi-station, no aux data |
| SVR — Li et al. (2017) | 22.04 | Single-station baseline |
| ARMA — Li et al. (2017) | 24.40 | Classical single-station |

> ✅ Our RMSE of **24.55** aligns with the paper's single-station baselines (ARMA: 24.40, SVR: 22.04), confirming that LSTMs are competitive with classical methods under equivalent single-station constraints.

### ⚙️ Setup & Run

```bash
# Install dependencies
pip install tensorflow scikit-learn pandas numpy matplotlib

# Run the notebook
jupyter notebook lstm_pm25/lstm_pm25.ipynb
```

---

## 📁 Project 2 — CNN-Based Lung Cancer Classification

### 🎯 Objective
Classify CT scan images into four categories — **Adenocarcinoma**, **Squamous Cell Carcinoma**, **Large Cell Carcinoma**, and **Normal** — using two CNN approaches: a custom architecture (MiniConvNet) and transfer learning (MobileNetV2).

### 📊 Dataset
- **Source:** [Chest CT-Scan Images — Kaggle](https://www.kaggle.com/datasets/mohamedhanyyy/chest-ctscan-images)
- **Reference Paper:** *Chowdhury et al. — "A lightweight CNN for enhanced non-small cell lung cancer classification using CT scan image," Nature Scientific Reports, 2026*

| Class | Train | Validation | Test | Total |
|-------|-------|------------|------|-------|
| Normal | 148 | 13 | 54 | 215 |
| Adenocarcinoma | 195 | 23 | 120 | 338 |
| Squamous Cell Carcinoma | 155 | 15 | 90 | 260 |
| Large Cell Carcinoma | 115 | 21 | 51 | 187 |
| **Total** | **613** | **72** | **315** | **1,000** |

### 🏗️ Model Architectures

**Model 1 — MiniConvNet (From Scratch)**
- 4 Convolutional blocks (Conv2D → BatchNorm → MaxPool)
- Filter count doubles per block starting from 16
- Input: `224×224×3`
- **Optimizer:** Adam | **Loss:** Categorical Cross-Entropy | **Epochs:** 30 | **Batch size:** 16

**Model 2 — Transfer Learning (MobileNetV2)**
- MobileNetV2 pretrained on ImageNet (base frozen)
- Custom head: `GlobalAvgPool → Dense(128) → Dropout(0.5) → Dense(4, Softmax)`
- **Optimizer:** Adam (`lr=0.0001`) | **Epochs:** 10 per phase | **Batch size:** 32

**Augmentation applied (training only):**
- Horizontal & vertical flip
- Random rotation ±20°
- Zoom range 0.2
- Brightness adjustment 0.8–1.2

### 📈 Results

| Model | Test Accuracy | Test Loss |
|-------|:------------:|:---------:|
| MiniConvNet (Ours — from scratch) | 44.1% | 1.41 |
| **MobileNetV2 Transfer Learning (Ours)** | **68.6%** | **0.80** |
| MiniConvNet — Paper (Chowdhury et al.) | 96% | — |

> ⚠️ The accuracy gap vs. the paper is primarily due to a **different dataset split** (613 train images vs. ~720 in the paper) and CPU-only training. Transfer learning significantly outperformed the scratch model.

### ⚙️ Setup & Run

```bash
# Install dependencies
pip install tensorflow scikit-learn pandas numpy matplotlib pillow

# Run notebooks
jupyter notebook cnn_lung_cancer/miniconvnet.ipynb
jupyter notebook cnn_lung_cancer/mobilenetv2_transfer.ipynb
```

---

## 🛠️ General Requirements

```
Python       >= 3.10
TensorFlow   >= 2.15
scikit-learn >= 1.0
pandas       >= 1.5
numpy        >= 1.23
matplotlib   >= 3.6
Pillow       >= 9.0
jupyter      >= 1.0
```

Install all at once:

```bash
pip install tensorflow scikit-learn pandas numpy matplotlib pillow jupyter
```

> 💡 **Recommended:** Run on Google Colab with a GPU runtime for significantly faster training, especially for the CNN models.

---

## 📚 References

1. X. Li et al., *"Long short-term memory neural network for air pollutant concentration predictions: Method development and evaluation,"* Environmental Pollution, vol. 231, pp. 997–1004, 2017.
2. S. Hochreiter and J. Schmidhuber, *"Long short-term memory,"* Neural Computation, vol. 9, no. 8, pp. 1735–1780, 1997.
3. Beijing PM2.5 Dataset — [Kaggle](https://www.kaggle.com/datasets/djhavera/beijing-pm25-data-data-set)
4. M.E.H. Chowdhury et al., *"A lightweight CNN for enhanced non-small cell lung cancer classification using CT scan image,"* Nature Scientific Reports, 2026.
5. Chest CT-Scan Images Dataset — [Kaggle](https://www.kaggle.com/datasets/mohamedhanyyy/chest-ctscan-images)

---
