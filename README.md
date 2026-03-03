# ================================================
# END-TO-END MLOPS PIPELINE WITH TFX
# Customer Churn Prediction (Local Execution)
# ================================================

# Project Objective
# ------------------------------------------------
# * Build a reproducible ML pipeline
# * Perform automated data validation
# * Apply feature engineering
# * Train a churn classification model
# * Save the trained model for serving
# * Track artifacts and metadata


# Tech Stack
# ------------------------------------------------
# * Python 3.10
# * TensorFlow 2.12
# * TensorFlow Extended (TFX 1.14.0)
# * TensorFlow Transform
# * TensorFlow Model Analysis
# * Apache Beam
# * Kubeflow-style DAG orchestration (Local)


# Project Structure
# ------------------------------------------------
# MLOps/
# ├── pipeline.py        -> Assembles full TFX pipeline
# ├── model.py           -> Keras churn classification model
# ├── transform.py       -> Feature engineering logic
# ├── requirements.txt   -> Dependencies
# └── data/
#     └── churn.csv      -> Telco Customer Churn dataset


# TFX Pipeline Components
# ------------------------------------------------
# 1. CsvExampleGen    -> Reads raw CSV data
# 2. StatisticsGen    -> Computes dataset statistics
# 3. SchemaGen        -> Infers feature schema
# 4. ExampleValidator -> Detects anomalies in data
# 5. Transform        -> Applies preprocessing & feature scaling
# 6. Trainer          -> Trains churn classification model
# 7. Pusher           -> Saves trained model for serving


# Feature Engineering (transform.py)
# ------------------------------------------------
# * Scales numeric features:
#     - tenure
#     - MonthlyCharges
#     - TotalCharges
#
# * Converts label:
#     - "Yes" -> 1
#     - "No"  -> 0


# Model Architecture (model.py)
# ------------------------------------------------
# Input -> Dense(16, ReLU)
#       -> Dense(8, ReLU)
#       -> Dense(1, Sigmoid)
#
# Loss      : Binary Crossentropy
# Optimizer : Adam
# Metric    : Accuracy


# Setup Instructions
# ------------------------------------------------
# 1. Clone repository
#    git clone <your-repo-link>
#    cd MLOps
#
# 2. Create virtual environment
#    py -3.10 -m venv venv
#    venv\Scripts\activate
#
# 3. Install dependencies
#    pip install -r requirements.txt
#
# 4. Run pipeline
#    python pipeline.py


# Output
# ------------------------------------------------
# * pipeline_output/  -> Contains TFX artifacts & metadata
# * serving_model/    -> Final trained model


# What This Project Demonstrates
# ------------------------------------------------
# * Modular ML pipeline design
# * Automated data validation
# * Reproducible training workflow
# * Production-style artifact tracking
# * End-to-end ML lifecycle automation
# * Practical MLOps implementation


# Author
# ------------------------------------------------
# Banoth Vasu
# Machine Learning & MLOps Enthusiast
