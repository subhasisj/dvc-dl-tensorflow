import os
import joblib
import logging
import tensorflow as tf
from src.utils.utils import get_timestamp


def create_save_tensorboard_callbacks(callbacks_dir,tensorboard_logs_dir): 

    unique_name = get_timestamp("tensorboard_logs")
    current_tb_log_dir = os.path.join(callbacks_dir,unique_name)
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=current_tb_log_dir)

    tensorboard_callback_file = os.path.join(callbacks_dir,unique_name + ".callback")
    joblib.dump(tensorboard_callback, tensorboard_callback_file)
    logging.info(f"Tensorboard Callbacks written to {tensorboard_callback_file}")

def create_save_checkpoint_callbacks():
    ...
