# 🔋 Lithium-Ion Battery SOH & RUL Prediction

This repository contains a deep learning pipeline using Long Short-Term Memory (LSTM) networks to predict the **State of Health (SOH)** and **Remaining Useful Life (RUL)** of lithium-ion batteries.

The project applies time-series sequence modeling to cycle-level charge and discharge data, a core requirement for modern Battery Management Systems (BMS) in Electric Vehicles (EVs).

## 🔍 Key Highlights

- 680 samples from three lithium-ion batteries (B5, B6, B7) with cycle-level degradation data.
- 10-cycle sliding windows over standardized charge/discharge, voltage, temperature, and capacity features.
- Separate LSTM regressors for SOH and RUL built in PyTorch.
- SOH model: final validation MSE ≈ 11 (RMSE ≈ 3.3 in normalized SOH units).
- RUL model: final validation MSE ≈ 0.6 (RMSE ≈ 0.8 in normalized RUL units).

## 📊 Dataset

The model is trained on the Kaggle [Lithium-ion battery degradation dataset](https://www.kaggle.com/datasets/programmer3/lithium-ion-battery-degradation-dataset) (inspired by the NASA Ames prognostics data).

- Contains 680 samples across 3 batteries (B5, B6, B7).
- **Features extracted**: Charge/Discharge Current (`chI`, `disI`), Voltage (`chV`, `disV`), Temperature (`chT`, `disT`), Cycle Index, and a capacity-like metric (`BCt`).

## 📁 Project Structure

```bash
battery_life_cycle_nasa/
├── battery_life_cycle.ipynb
├── Battery_dataset.csv
├── requirements.txt
├── README.md
└── images/
    ├── soh-training.png
    ├── soh-true-vs-pred-2.png
    ├── soh-eval-4.png
    └── data-example.png
```

## 🧠 Methodology & Architecture

The data is standardized and windowed into **10-cycle sequences** to capture long-term degradation trends rather than just instantaneous state.

Two isolated models were trained for the distinct prediction tasks:

1. **SOH Prediction Model**:
   - **Architecture**: 1-layer LSTM (Hidden Size: 100) → Linear Layer
   - **Performance**: Converged with a final validation MSE ≈ `11`, which translates to an RMSE of roughly `3.3%` in SOH tracking.
2. **RUL Prediction Model**:
   - **Architecture**: 1-layer LSTM (Hidden Size: 150) → Linear Layer
   - **Performance**: Converged with a final validation MSE of `0.6` on normalized RUL targets (RMSE ≈ `0.8`).

## 📈 Results

The models demonstrate strong convergence and accurately map the non-linear degradation curves of the lithium-ion cells.

### SOH Prediction Convergence

![SOH Training Curve](images/soh-training.png)

### True vs Predicted SOH Scatter

![SOH Scatter Plot](images/soh-true-vs-pred-2.png)

### SOH Evaluation Across Cycles

![SOH Time-Series](images/soh-eval-4.png)

### Battery Degradation Profile Example

![Battery Profile Example](images/data-example.png)

## ⚙️ How to Run

1. Clone this repository:
   ```bash
   git clone https://github.com/abhishekspawar0803-svg/battery_life_cycle_nasa.git
   cd battery_life_cycle_nasa
   ```
2. Install dependencies:
   ```bash
   pip install torch pandas numpy matplotlib scikit-learn
   ```
3. Download the dataset from Kaggle and place it in the project root.
4. Open the Jupyter Notebook (`.ipynb`) and execute the cells sequentially.

## 🧩 Applications

- **Battery Management Systems (BMS)** – Provides data-driven SOH and RUL estimates that can be integrated into EV BMS pipelines for smarter charge/discharge control and cell balancing.
- **Predictive maintenance** – Enables early detection of abnormal degradation and scheduling of pack/service interventions before failure.
- **Lifecycle-aware control** – Allows energy management strategies (e.g., fast charging, power limits) to adapt based on the predicted remaining life instead of fixed safety margins.

## 🚀 Future Scope

- **On-board deployment** – Quantize and deploy the trained LSTM models onto embedded hardware (e.g., ESP32 or automotive-grade MCUs) for real-time SOH/RUL estimation inside a BMS.
- **Hybrid physics–ML models** – Extend the approach with physics-informed neural networks (PINNs) or equivalent circuit model constraints to improve extrapolation beyond the observed degradation range.
- **Multi-cell and pack-level modeling** – Scale from single-cell predictions to module/pack-level health estimation, including imbalance detection and cell-to-cell variation.
- **Richer feature engineering** – Incorporate additional features such as dV/dt, dI/dt, or derived aging indicators to capture subtler degradation mechanisms.
