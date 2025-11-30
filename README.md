📘 Publication Context

This repository contains the dataset, source code, and Digital Twin simulation framework for the book chapter:

"Case Study: AI-Driven Acoustic Surveillance for Water Infrastructure Resilience"

This work is a contribution to the upcoming book:

Title: AI-IoT for Smart Urban Planning and Sustainable Development Publisher: Springer Nature (Urban Sustainability Series) Editors: Dr. Abdulaziz I. Almulhim, Prof. Ayyoob Sharifi, Prof. Baojie He, Prof. Ali Cheshmehzangi, and Prof. Martin de Jong.

🏙️ About the Book

This book explores the transformative role of Artificial Intelligence (AI) and the Internet of Things (IoT) in urban planning. It addresses the critical need for innovative solutions that leverage technology to create sustainable urban infrastructures, focusing on energy efficiency, transportation systems, and water management.

📑 Chapter Abstract

Aligned with Section II: AI-IoT Applications in Sustainable Urban Infrastructure (specifically Water Management and Smart Cities), this chapter presents "Hydro-Twin." It is a hybrid Deep Learning framework (CNN-LSTM) designed to detect and classify leak anomalies in water distribution networks, moving utility management from reactive repair to proactive, real-time condition monitoring.

![Status](https://img.shields.io/badge/Status-Active-success)
![Python](https://img.shields.io/badge/Python-3.10-blue)
![Framework](https://img.shields.io/badge/Framework-Flask%20%7C%20TensorFlow-orange)

## 📖 Overview

**Smart-Hydro-Twin** is a hybrid Deep Learning framework and 3D Digital Twin designed to detect and classify leak anomalies in urban water distribution networks. By processing acoustic signals from hydrophone sensors, it identifies specific failure modes (e.g., Gasket Leaks, Cracks) and visualizes them in a real-time web dashboard.

This repository contains the complete source code for the **CNN-LSTM AI model**, the **Physics Engine**, and the **3D Interface**.

## 🧠 The AI Architecture

The system uses a hybrid neural network to analyze chaotic acoustic data:
1.  **CNN (The Ear):** 1D Convolutional layers extract spatial features from raw audio waveforms.
2.  **LSTM (The Brain):** Captures temporal dependencies to distinguish between sustained leaks and transient noise.

**Classifications:**
* **NL:** No Leak (Normal)
* **GL:** Gasket Leak
* **LC:** Longitudinal Crack
* **CC:** Circumferential Crack
* **OL:** Orifice Leak

## 📂 Repository Structure

```text
hydro-twin-ai/
├── training/           # Python scripts for model training
│   └── train_model.py  # Run this to regenerate model.keras
├── web_app/            # The Digital Twin Interface
│   ├── templates/
│   │   └── index.html  # Three.js Visualization
│   └── app.py          # Flask Backend & Physics Engine
├── requirements.txt    # Dependencies
└── README.md           # Documentation
