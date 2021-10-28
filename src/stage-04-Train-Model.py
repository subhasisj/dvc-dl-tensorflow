import argparse
import io
import logging
import os

import pandas as pd
from tqdm import tqdm

from src.utils.utils import create_directory, read_yaml

from src.utils.callbacks import save_tensorboard_callbacks, save_checkpoint

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
    

if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config", "-c", default="config/config.yaml")
    args.add_argument("--params", "-p", default="params.yaml")
    parsed_args = args.parse_args()

    try:
        logging.info(" >>>>>> Starting Stage 04 - Training Started")
        train_model(parsed_args.config, parsed_args.params)
        logging.info("Stage 04 Training completed successfully >>>>>>")
    except Exception as e:
        logging.exception(e)
        raise e
