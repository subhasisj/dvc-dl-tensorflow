import os
import joblib
import logging
import tensorflow as tf
from src.utils.utils import get_timestamp


def save_tensorboard_callbacks(callbacks_dir, tensorboard_logs_dir):

    unique_name = get_timestamp("tensorboard_logs")
    tensorboard_logs_dir = os.path.join(callbacks_dir, unique_name)
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=tensorboard_logs_dir)

    tensorboard_callback_file = os.path.join(callbacks_dir, unique_name + ".callback")
    joblib.dump(tensorboard_callback, tensorboard_callback_file)
    logging.info(f"Tensorboard Callbacks written to {tensorboard_callback_file}")


def save_checkpoint(callbacks_dir,checkpoint_dir):
    checkpoint_file_path = os.path.join(checkpoint_dir,"ckpt_model.h5")
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_file_path,
        save_best_only=True,
        verbose=1,
    )

    checkpoint_callback_file = os.path.join(callbacks_dir, "checkpoint.callback")
    joblib.dump(checkpoint_callback, checkpoint_callback_file)
    logging.info(f"Checkpoint Callbacks written to {checkpoint_callback_file}")
