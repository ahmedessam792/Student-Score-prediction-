# 🎓  Student Exam Score Predictor

**Predict student exam performance using machine learning** – a full end-to-end ML web app built with **Streamlit**, **scikit-learn**, and **pandas**.



---

## Project Overview

This project uses **Linear Regression** and **Polynomial Regression (Degree 2)** to predict a student's **exam score** based on **19 academic, behavioral, and socioeconomic factors**.

The **best model is automatically selected** by comparing **R² score**.

The **interactive Streamlit app** lets users input student details and get **instant predictions**.

---


## Model Performance (Final Results)

| Model                  | **MSE** | **RMSE** | **MAE** | **R²** |
|------------------------|--------|---------|--------|-------|
| **Linear Regression**  | 0.9428 | 0.9710  | 0.5643 | 0.9164 |
| **Polynomial (Deg 2)** | 0.6783 | 0.8236  | 0.3843 | 0.9398 |

> **Best Model:** **Polynomial Regression (Degree 2)**  
> Explains **93.98%** of the variance in exam scores.

---

## Input Features

| Feature | Description | Type | Example |
|--------|-------------|------|---------|
| `Hours_Studied` | Number of hours studied per day | **Numeric** | `20` |
| `Attendance` | Attendance percentage in class | **Numeric** | `90` |
| `Sleep_Hours` | Average hours of sleep per night | **Numeric** | `7` |
| `Previous_Scores` | Scores in previous exams | **Numeric** | `75` |
| `Tutoring_Sessions` | Number of tutoring sessions attended | **Numeric** | `2` |
| `Physical_Activity` | Hours of physical activity per week | **Numeric** | `3` |
| `Motivation_Level` | Student’s self-reported motivation | **Categorical** | `High`, `Medium`, `Low` |
| `Parental_Involvement` | Level of parental support in studies | **Categorical** | `High`, `Medium`, `Low` |
| `Access_to_Resources` | Availability of books, internet, etc. | **Categorical** | `High`, `Medium`, `Low` |
| `Family_Income` | Family income bracket | **Categorical** | `High`, `Medium`, `Low` |
| `Teacher_Quality` | Perceived quality of teaching | **Categorical** | `High`, `Medium`, `Low` |
| `Distance_from_Home` | How far the student lives from school | **Categorical** | `Near`, `Moderate`, `Far` |
| `Peer_Influence` | Influence of peers on study habits | **Categorical** | `Positive`, `Neutral`, `Negative` |
| `Gender` | Gender of the student | **Categorical** | `Male`, `Female` |
| `School_Type` | Type of school | **Categorical** | `Public`, `Private` |
| `Internet_Access` | Availability of internet at home | **Categorical** | `Yes`, `No` |
| `Extracurricular_Activities` | Participation in clubs/sports | **Categorical** | `Yes`, `No` |
| `Learning_Disabilities` | Diagnosed learning challenges | **Categorical** | `Yes`, `No` |
| `Parental_Education_Level` | Highest education of parents | **Categorical** | `Postgraduate`, `College`, `High School` |

---

## Key Features

- Real-time prediction via **Streamlit**
- Automatic **model selection** (Linear vs Polynomial)
- Full preprocessing: **outlier capping**, **encoding**, **scaling**, **polynomial features**
- Model & pipeline saved with `joblib` for production

---

## Tech Stack

| Category           | Technologies                          |
|--------------------|---------------------------------------|
| **Language**       | Python 3.11+                          |
| **Framework**      | Streamlit                             |
| **ML Libraries**   | scikit-learn, pandas, numpy           |
| **Visualization**  | matplotlib, seaborn                   |
| **Deployment**     | Streamlit Community Cloud             |
---
## Model Pipeline

```mermaid
flowchart TD
    A["Raw Input"] --> B["Handle Missing Values"]
    B --> C["Outlier Capping (IQR)"]
    C --> D["Ordinal Encoding"]
    D --> E["One-Hot Encoding"]
    E --> F["Standard Scaling"]
    F --> G{"Use Polynomial?"}
    G -->|Yes| H["Polynomial Features (Deg 2)"]
    G -->|No| I["Linear Regression"]
    H --> J["Polynomial Regression"]
    I --> K["Prediction"]
    J --> K
