import tensorflow as tf
import os
import logging


def get_vgg16(input_shape, model_path):
    model = tf.keras.applications.vgg16.Vgg16(
        input_shape=input_shape, weights="imagenet", include_top=False
    )

    model.save(model_path)
    logging.info(f"Model saved to {model_path}")
    return model