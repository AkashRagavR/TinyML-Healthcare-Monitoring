### TinyML-Based Smart Healthcare Monitoring System
### Performance Profiling of TinyML-Enabled IoT Nodes for Edge-Based Heart Disease Prediction

## Overview

This project implements a fully on-device (edge) heart disease monitoring
system that runs entirely on low-power microcontrollers with no cloud
dependency. It combines two machine learning models:

- A 1D CNN for ECG signal classification (5 classes: Normal,
  Supraventricular, Ventricular, Fusion, Unclassifiable)
- An XGBoost model for vital sign classification (Normal / Abnormal),
  using heart rate, SpO2, and body temperature

Both models are converted into TinyML-compatible formats and deployed
across three different microcontroller boards to benchmark performance:

- ESP32
- Arduino Nano 33 BLE
- Sony Spresense

Sensors used: AD8232 (ECG), MAX30102 (pulse rate + SpO2), MLX90614
(non-contact infrared body temperature).

The system reads live sensor data, runs both models on-device, and
combines their outputs into a single patient condition status
(Normal/Abnormal), along with performance metrics: ML inference time,
total latency, flash memory usage, power consumption, and energy per
inference.

## Repository Structure

TinyML-Healthcare-Monitoring/
│
├── Machine Learning/          # Model training and dataset (Python/Jupyter)
│   ├── CNN.ipynb              # CNN training notebook for ECG classification
│   ├── XGB.ipynb              # XGBoost training + export_to_c() notebook
│   ├── mitbih_train.csv       # MIT-BIH arrhythmia training dataset
│   ├── mitbih_test.csv        # MIT-BIH arrhythmia test dataset
│   └── Human_vital_signs_R.csv # Vital sign dataset (HR, SpO2, Temp)
│
├── Esp32/                      # ESP32 deployment
│   ├── Final_esp32.ino
│   ├── model_data.cc           # CNN model baked as C byte array
│   └── xgb_model.h             # XGBoost model as C if-else tree logic
│
├── Arduino Nano33 ble/         # Arduino Nano 33 BLE deployment
│   ├── Nano.ino
│   ├── model_data.cc
│   └── xgb_model.h
│
└── Sony-Spresense/             # Sony Spresense deployment
    ├── Spresense.ino
    ├── model_data.cc
    └── xgb_model.h
```

Note: `model_data.cc` and `xgb_model.h` are identical across all three
boards — only the sketch (.ino) and the TFLite library import differ per
board, so results reflect hardware performance, not model variation.

## Hardware Requirements

----------------------------------------------------------------
| Component               | Purpose                            |
|-------------------------|------------------------------------|
| ESP32                   | Target board                       |
| Arduino Nano 33 BLE     | Target board                       |
| Sony Spresense          | Target board                       |
| AD8232                  | ECG signal acquisition             |
| MAX30102                | Heart rate + SpO2 (pulse oximeter) |
| MLX90614                | Non-contact body temperature (I2C) |
----------------------------------------------------------------

## Software / Libraries
--------------------------------------------------------------
| Board               | TFLite Micro Library                 |
|---------------------|--------------------------------------|
| ESP32-S3            | TensorFlowLite_ESP32                 |
| Arduino Nano 33 BLE | TensorFlow Lite for Microcontrollers |
| Sony Spresense      | Chirale_TensorFlowLite               |
--------------------------------------------------------------


XGBoost inference requires no external library — it runs as plain C
code (`xgb_model.h`) generated from the trained model.

## Models

### 1. CNN — ECG Classification
- Input: 187-sample ECG window
- Architecture: Conv1D (8 filters, kernel 3) → MaxPooling1D → Flatten →
  Dense(16) → Dense(5, Softmax)
- Trained on MIT-BIH Arrhythmia dataset (21,892 samples), 5 epochs
- Test Accuracy: 93.99% | Weighted F1: 0.93
- Converted to INT8-quantized `.tflite`, then exported to `model_data.cc`

### 2. XGBoost — Vital Sign Classification
- Inputs: HR, SpO2, Temperature + 12 engineered features (ratios,
  squared terms, clinical risk flags) → 15 total features
- Config: 50 estimators, max_depth 4, learning_rate 0.1
- Test Accuracy: 99.27%
- Exported directly to C via a custom `export_to_c()` script → `xgb_model.h`

## How to Run

1. Open the `Machine Learning/` notebooks to retrain or inspect the
   models (CNN.ipynb, XGB.ipynb).
2. Pick the target board folder (`Esp32/`, `Arduino Nano 33 ble/`, or
   `Sony Spresense/`) and open the `.ino` file in Arduino IDE.
3. Install the board-specific TFLite Micro library listed above.
4. Wire up the AD8232, MAX30102, and MLX90614 sensors.
5. Compile and flash the sketch to the board.
6. Open the Serial Monitor to view: ECG class, vital sign status,
   combined patient condition, and performance metrics
   (inference time, latency, flash used, power, energy/inference).

## Results Summary

-------------------------------------------------------------------------
| Metric                      | ESP32-S3 | Nano 33 BLE | Sony Spresense |
|-----------------------------|----------|-------------|----------------|
| CNN Inference Time (us)     | 8051     | 5000        | 6000           |
| XGBoost Inference Time (us) | 192      | 61          | 30             |
| Total Latency (us)          | 8243     | 5061        | 6030           |
| Flash Used (KB)             | 557      | 768         | 662            |
| Power Consumption (mA)      | 110      | 80          | 60             |
| Energy per Inference (uJ)   | 907      | 405         | 362            |
-------------------------------------------------------------------------


Arduino Nano 33 BLE gave the fastest CNN inference; Sony Spresense gave
the best energy efficiency and lowest power draw, making it the most
suitable platform for battery-powered wearable deployment. ESP32-S3
consumed more power but offers built-in wireless connectivity.

## Future Work

- Model quantization and pruning for further latency/energy reduction
- Hardware-accelerated inference
- Long-term wearable form-factor testing

## License

Add a license of your choice (e.g., MIT) before making the repo public.
