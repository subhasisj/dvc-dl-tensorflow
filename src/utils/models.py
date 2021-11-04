from re import M
import tensorflow as tf
import os
import logging
from src.utils.utils import get_timestamp


def get_vgg16(input_shape, model_path):
    model = tf.keras.applications.vgg16.VGG16(
        input_shape=input_shape, weights="imagenet", include_top=False
    )

    model.save(model_path)
    logging.info(f"Model saved to {model_path}")
    return model


def prepare_model(
    model,
    classes,
    freeze_till,
    learning_rate,
    freeze_all=False,
):

    if freeze_all:
        for layer in model.layers:
            layer.trainable = False

    elif (freeze_till is not None) and (freeze_till > 1):
        for layer in model.layers[:-freeze_till]:
            layer.trainable = False

    flatten_in = tf.keras.layers.Flatten()(model.output)
    prediction = tf.keras.layers.Dense(units=classes, activation="softmax")(flatten_in)

    full_model = tf.keras.models.Model(inputs=model.input, outputs=prediction)

    full_model.compile(
        optimizer=tf.keras.optimizers.SGD(learning_rate=learning_rate),
        loss=tf.keras.losses.CategoricalCrossentropy(),
        metrics=["accuracy"],
    )

    logging.info("Model compiled Successfully.")

    return full_model


def load_pretrained_model(pretrained_model_path):
    # Load model
    model = tf.keras.models.load_model(pretrained_model_path)
    logging.info(f"Model loaded from {pretrained_model_path}")
    return model


def get_unique_model_file_path(finetuned_model_dir, model_name="model"):

    return os.path.join(finetuned_model_dir, f"{get_timestamp(model_name)}.h5")
