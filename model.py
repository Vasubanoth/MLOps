"""
model.py — Churn Classification Model
"""

import tensorflow as tf
import tensorflow_transform as tft
from tfx.components.trainer.fn_args_utils import FnArgs
from transform import NUMERIC_FEATURES, LABEL_KEY, transformed_name


TRANSFORMED_FEATURES = [transformed_name(f) for f in NUMERIC_FEATURES]


def build_model():
    inputs = {
        name: tf.keras.Input(shape=(1,), name=name)
        for name in TRANSFORMED_FEATURES
    }

    x = tf.keras.layers.concatenate(list(inputs.values()))

    x = tf.keras.layers.Dense(16, activation="relu")(x)
    x = tf.keras.layers.Dense(8, activation="relu")(x)

    output = tf.keras.layers.Dense(1, activation="sigmoid")(x)

    model = tf.keras.Model(inputs=inputs, outputs=output)

    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    return model


def run_fn(fn_args: FnArgs):

    tf_transform_output = tft.TFTransformOutput(fn_args.transform_output)

    train_dataset = tf.data.experimental.make_batched_features_dataset(
        file_pattern=fn_args.train_files,
        batch_size=32,
        features=tf_transform_output.transformed_feature_spec(),
        label_key=transformed_name(LABEL_KEY)
    )

    eval_dataset = tf.data.experimental.make_batched_features_dataset(
        file_pattern=fn_args.eval_files,
        batch_size=32,
        features=tf_transform_output.transformed_feature_spec(),
        label_key=transformed_name(LABEL_KEY)
    )

    model = build_model()

    model.fit(train_dataset, epochs=5)

    model.save(fn_args.serving_model_dir, save_format="tf")
