![CSAT Banner](https://images.unsplash.com/photo-1556155092-8707de31f9c4?q=80\&w=1600\&auto=format\&fit=crop)

# CSAT Prediction System — Hybrid Deep Learning Ensemble

**End-to-End Production-Grade Machine Learning System for Customer Satisfaction (CSAT) Prediction**
Built with deep learning, intelligent ensembling, business-aware error control, and interactive deployment.

---

## Project Overview

This project implements a **full-stack ML system** for predicting **Customer Satisfaction Score (CSAT: 1–5)** using a **hybrid deep learning ensemble architecture**.

Unlike typical notebook-only ML projects, this system covers the **entire ML lifecycle**:

* Data ingestion and cleaning
* Feature engineering
* Preprocessing pipelines
* Multi-model deep learning training
* Intelligent ensemble inference
* Business-aware error handling
* Model evaluation framework
* Interactive deployment (Streamlit)
* Model governance and metrics tracking

It is designed as a **real-world ML product system**, not a demo notebook.

---

## Problem Statement

Customer satisfaction directly impacts:

* Customer retention
* Brand trust
* Revenue growth
* Operational optimization

Traditional CSAT analysis is reactive. This system enables **predictive CSAT intelligence**, allowing organizations to:

* Proactively identify dissatisfied customers
* Prioritize high-risk interactions
* Optimize support workflows
* Reduce churn risk
* Improve service quality using predictive insights

---

## System Architecture

Raw Data → Cleaning → Feature Engineering → Preprocessing Pipeline → Deep Learning Models → Hybrid Ensemble → Prediction Engine → Evaluation Layer → Streamlit Dashboard

---

## Data Engineering

### Cleaning Pipeline

* Column pruning
* Invalid value handling
* Null value treatment
* Datetime normalization
* Median-based imputation
* Type sanitization

### Feature Engineering

Temporal features:

* response_time_minutes
* log_response_time
* hour_reported
* day_of_week
* month
* is_weekend
* is_business_hours

Agent-level intelligence features:

* agent_avg_csat
* agent_case_count

---

## Preprocessing Pipeline

A persistent preprocessing object (`preprocessor.pkl`) is used for:

* Encoding categorical variables
* Scaling numerical features
* Feature transformations
* Consistent inference-time processing

---

## Model Architecture

### Hybrid Deep Learning Ensemble

Three complementary neural architectures:

1. Wide + Shallow Network
2. Deep + Narrow Network
3. Balanced Architecture (Swish Activation)

---

## Ensemble Intelligence Layer

* Soft-voting via probability averaging
* Multi-model confidence aggregation
* Probability sanitization
* Distribution normalization

### Business-Aware Override Logic

Special handling for critical business classes:

* CSAT 1 (Highly dissatisfied)
* CSAT 5 (Highly satisfied)

---

## Prediction Output

* Final CSAT class (1–5)
* Class probability distribution
* Confidence score
* Error risk category

---

## Evaluation Framework

* Overall accuracy
* Per-class accuracy
* Confusion matrix
* Critical error tracking
* Balanced class evaluation

---

## Interactive Deployment (Streamlit)

Modules:

Home
Single Prediction
Batch Testing
Model Performance
About

---

## Tech Stack

* Python
* Pandas, NumPy
* TensorFlow / Keras
* Scikit-learn
* Joblib
* Streamlit
* Plotly

---

## Project Structure

```
project/
│
├── csat_dashboard.py
├── csat_prediction_final.ipynb
├── preprocessor.pkl
├── ensemble_model_1_wide_shallow.h5
├── ensemble_model_2_deep_narrow.h5
├── ensemble_model_3_balanced_swish.h5
├── model_metrics.json
├── eCommerce_Customer_support_data.csv
└── README.md
```

---

## Setup Instructions

```bash
git clone <repo_url>
cd <repo_name>
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
streamlit run csat_dashboard.py
```

---

## Business Value

* Predict dissatisfaction before escalation
* Reduce churn probability
* Optimize support operations
* Prioritize high-risk customers
* Improve service quality

---

## Author

**Archit Dhodi**

---

## License

Academic, learning, and portfolio demonstration use.
