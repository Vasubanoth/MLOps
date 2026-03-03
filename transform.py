"""
transform.py — Feature Engineering for Customer Churn
"""

import tensorflow as tf
import tensorflow_transform as tft

# Numeric features from Telco dataset
NUMERIC_FEATURES = [
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
]

# Target column
LABEL_KEY = "Churn"

def transformed_name(key: str) -> str:
    return key + "_xf"


def preprocessing_fn(inputs: dict) -> dict:
    outputs = {}

    # Scale numeric features
    for feature in NUMERIC_FEATURES:
        outputs[transformed_name(feature)] = tft.scale_to_z_score(
            tf.cast(inputs[feature], tf.float32)
        )

    # Convert label (Yes/No → 1/0)
    outputs[transformed_name(LABEL_KEY)] = tf.cast(
        tf.equal(inputs[LABEL_KEY], "Yes"), tf.int64
    )

    return outputs
