# Student Exam Score Predictor

**Predict exam scores with machine learning** – an end-to-end web app built with **Streamlit**, **scikit-learn**, and **pandas**.

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-app-link.streamlit.app)
[![Python 3.11+](https://img.shields.io/badge/Python-3.11+-blue)](#)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](#)

---

## Live Demo

**Try it now:**  
[Open the App](https://your-app-link.streamlit.app)

---

## Project Overview

Predicts a student’s **Exam Score** from **19 academic & personal factors** using **Linear Regression** and **Polynomial Regression (Degree 2)**.  
The **best model (R² = 0.9398)** is automatically selected and deployed via Streamlit.

---

## Model Performance

| Model                  | MSE   | RMSE  | MAE   | **R²** |
|------------------------|-------|-------|-------|--------|
| Linear Regression      | 0.9428| 0.9710| 0.5643| **0.9164** |
| **Polynomial (Deg 2)** | **0.6783**| **0.8236**| **0.3843**| **0.9398** |

> **Best Model:** **Polynomial Regression (Degree 2)** – explains **93.98 %** of variance.

---

## Model Pipeline (Mermaid)

```mermaid
flowchart TD
    A[Raw Input] --> B[Handle Missing Values]
    B --> C[Outlier Capping (IQR)]
    C --> D[Ordinal Encoding]
    D --> E[One-Hot Encoding]
    E --> F[Standard Scaling]
    F --> G{Use Polynomial?}
    G -->|Yes| H[Polynomial Features (Deg 2)]
    G -->|No| I[Linear Regression]
    H --> J[Polynomial Regression]
    I --> K[Prediction]
    J --> K
