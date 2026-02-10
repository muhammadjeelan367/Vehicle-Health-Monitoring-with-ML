# 🚗 Vehicle Health Monitoring with Machine Learning

> An interactive **Streamlit web application** that predicts engine health conditions in real-time using a trained **Gradient Boosting Machine (GBM)** model — enabling proactive fleet maintenance and preventing costly breakdowns.

---

## 📋 Problem Statement

Vehicle engines degrade over time, and early detection of abnormal conditions can prevent **catastrophic failures**, **road accidents**, and **expensive emergency repairs**. Traditional monitoring relies on dashboard warning lights that often trigger *after* damage has already begun.

This project builds a **machine learning-powered monitoring system** that:
- Analyzes real-time sensor readings (RPM, pressures, temperatures)
- Predicts whether the engine is in a **normal** or **warning** state
- Provides a **confidence score** for each prediction
- Enables proactive maintenance scheduling **2–3 weeks in advance**

---

## 🏗️ Solution Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                   DATA PREPROCESSING                         │
│  engine_data.csv (19,535 records)                           │
│      │                                                      │
│      ▼                                                      │
│  Feature Engineering:                                       │
│  • Engine Power = RPM × Lub Oil Pressure                    │
│  • Temperature Difference = Coolant Temp - Lub Oil Temp     │
│      │                                                      │
│      ▼                                                      │
│  Train/Test Split (60/40) ──► GBM Classifier                │
│      │                                                      │
│      ▼                                                      │
│  Serialize Model ──► model.pkl                              │
└──────────────────────────┬──────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────┐
│                   STREAMLIT WEB APP                           │
│                                                              │
│  ┌──────────────────────────────────────────────────┐       │
│  │  Interactive Sliders for 7 Engine Parameters:     │       │
│  │  • Engine RPM (61 – 2,239)                       │       │
│  │  • Lub Oil Pressure (0.003 – 7.27)               │       │
│  │  • Fuel Pressure (0.003 – 21.14)                 │       │
│  │  • Coolant Pressure (0.003 – 7.48)               │       │
│  │  • Lub Oil Temperature (71.3°– 89.6°)            │       │
│  │  • Coolant Temperature (61.7° – 195.5°)          │       │
│  │  • Temperature Difference (-22.7° – 119.0°)      │       │
│  └──────────────────────┬───────────────────────────┘       │
│                         │                                    │
│                         ▼                                    │
│  ┌──────────────────────────────────────────────────┐       │
│  │  Prediction Output:                               │       │
│  │  ✅ "Engine in normal condition" + Confidence %    │       │
│  │  ⚠️ "Warning! Investigate further" + Confidence % │       │
│  └──────────────────────────────────────────────────┘       │
└──────────────────────────────────────────────────────────────┘
```

---

## 🧰 Tech Stack

| Category | Technologies |
|---|---|
| **Language** | Python |
| **ML Model** | Gradient Boosting Classifier (Scikit-learn) |
| **Data Processing** | Pandas, NumPy |
| **Web Framework** | Streamlit |
| **Model Serialization** | Pickle |
| **Environment** | Jupyter Notebook (training), Streamlit (deployment) |

---

## 📂 Project Structure

```
Vehicle-Health-Monitoring-with-ML/
├── app.py                        # Streamlit web application
├── data_preprocessing.ipynb      # Full EDA, feature engineering & model training
├── engine_data.csv               # Dataset (19,535 engine sensor records)
├── hhmodel.pkl                   # Trained GBM model (serialized)
├── Predictive maintenance PPT.pptx  # Project presentation
└── README.md
```

---

## ✨ Key Features

- **Interactive Streamlit Dashboard** — Adjust engine parameters via sliders and get instant predictions.
- **Gradient Boosting Classifier** — Robust ensemble model chosen for its ability to handle non-linear relationships in sensor data.
- **Confidence Scoring** — Each prediction includes a probability percentage using `predict_proba`, giving fleet managers actionable confidence levels.
- **Feature Engineering** — Derived `Temperature_difference` (Coolant Temp - Lub Oil Temp) to capture thermal stress patterns.
- **Real-Time Inference** — Load the pre-trained `.pkl` model and predict engine condition in milliseconds.
- **Sidebar Feature Guide** — Built-in descriptions for each sensor parameter, making the app accessible to non-technical users.

---

## 📊 Dataset Details

| Attribute | Value |
|---|---|
| **Records** | 19,535 engine sensor readings |
| **Features** | 7 (after engineering) |
| **Target** | Engine Condition (0 = Normal, 1 = Warning) |
| **Class Distribution** | ~63% Warning, ~37% Normal |

### Input Features

| Feature | Range | Description |
|---|---|---|
| Engine RPM | 61 – 2,239 | Revolutions per minute |
| Lub Oil Pressure | 0.003 – 7.27 | Lubricating oil pressure |
| Fuel Pressure | 0.003 – 21.14 | Fuel system pressure |
| Coolant Pressure | 0.003 – 7.48 | Cooling system pressure |
| Lub Oil Temp | 71.3° – 89.6° | Lubricating oil temperature |
| Coolant Temp | 61.7° – 195.5° | Coolant temperature |
| Temp Difference | -22.7° – 119.0° | Coolant Temp − Lub Oil Temp |

---

## 🧠 My Learning Journey

This project represents my journey from **understanding ML theory** to **building a user-facing application** that non-technical stakeholders can actually use.

| Area | What I Learned |
|---|---|
| **Feature Engineering** | Deriving `Temperature_difference` from raw sensor data — understanding that the *relationship between features* often matters more than the features themselves. My automotive service background helped me recognize that thermal stress (coolant vs. oil temperature delta) is a key failure indicator. |
| **Gradient Boosting** | Deep understanding of ensemble methods — how sequential weak learners correct each other's errors. Tuned hyperparameters (`n_estimators=100`, `learning_rate=0.1`, `max_depth=3`, `subsample=0.8`) to balance bias-variance. |
| **Model Deployment** | Moving from notebook experiments to a production-ready Streamlit app — serializing models with Pickle, designing intuitive UIs with parameter sliders, and displaying human-readable predictions with confidence scores. |
| **Confidence Calibration** | Using `predict_proba` instead of just `predict` — understanding that binary predictions lose valuable information, and probability outputs enable risk-based decision making for fleet managers. |
| **User Experience** | Designing the sidebar with feature descriptions so that mechanics and fleet managers (who aren't ML engineers) can understand what each slider means and how to interpret the results. |

---

## 🚀 Getting Started

### Prerequisites
- Python 3.8+

### Installation

```bash
# Clone the repository
git clone https://github.com/<your-username>/Vehicle-Health-Monitoring-with-ML.git
cd Vehicle-Health-Monitoring-with-ML

# Install dependencies
pip install -r requirements.txt
```

### Run the Web Application

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`.

### Retrain the Model

Open `data_preprocessing.ipynb` in Jupyter Notebook to:
1. Explore the dataset
2. Perform feature engineering
3. Train and evaluate the GBM model
4. Export the updated model as `hhmodel.pkl`

---

## 📈 Model Performance

| Metric | Class 0 (Normal) | Class 1 (Warning) | Overall |
|---|---|---|---|
| **Precision** | 0.59 | 0.69 | 0.65 (weighted) |
| **Recall** | 0.36 | 0.85 | 0.67 (weighted) |
| **F1-Score** | 0.45 | 0.76 | 0.64 (weighted) |
| **Accuracy** | — | — | **0.67** |

> **Note:** The model prioritizes **high recall for the warning class (0.85)**, which is the correct trade-off for predictive maintenance — it's better to investigate a false alarm than to miss a real failure.

---

## 📄 License

This project is open-source. Feel free to use, modify, and distribute.
