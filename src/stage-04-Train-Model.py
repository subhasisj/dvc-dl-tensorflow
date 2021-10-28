import argparse
import io
import logging
import os

import pandas as pd
from tqdm import tqdm
import tensorflow as tf

from src.utils.utils import create_directory, read_yaml
from src.utils.models import load_pretrained_model
from src.utils.callbacks import save_tensorboard_callbacks, save_checkpoint,get_callbacks

log_directory = os.path.join(os.getcwd(), "logs")
os.makedirs(log_directory, exist_ok=True)
logging_str = "[%(asctime)s: %(levelname)s: %(module)s:] %(message)s"
log_filename = "stage-logs.log"
logging.basicConfig(
    level=logging.INFO,
    format=logging_str,
    filename=os.path.join(log_directory, log_filename),
    filemode="a",
)


def train_model(path_to_config, path_to_params):
    config = read_yaml(path_to_config)

    artifacts_dir = config["artifacts"]["DIR"]

    TRAINED_MODEL_DIR = os.path.join(artifacts_dir, config["artifacts"]["TRAINED_MODEL_DIR"])
    create_directory([TRAINED_MODEL_DIR])

    pretrained_model_dir_path = os.path.join(artifacts_dir, config["artifacts"]["BASE_MODEL_DIR"])
    pretrained_model_path = os.path.join(pretrained_model_dir_path, config["artifacts"]["BASE_MODEL_NAME"])
    
    model = load_pretrained_model(pretrained_model_path)

    callbacks_dir = os.path.join(artifacts_dir, config["artifacts"]["CALLBACKS_DIR"])

    callbacks = get_callbacks(callbacks_dir)



if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config", "-c", default="config/config.yaml")
    args.add_argument("--params", "-p", default="params.yaml")
    parsed_args = args.parse_args()

    try:
        logging.info(" \n\n>>>>>> Starting Stage 04 - Training Started")
        train_model(parsed_args.config, parsed_args.params)
        logging.info("Stage 04 Training completed successfully >>>>>>\n\n")
    except Exception as e:
        logging.exception(e)
        raise e
