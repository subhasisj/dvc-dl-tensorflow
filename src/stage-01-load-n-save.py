import argparse
import logging
import os

import pandas as pd
from tqdm import tqdm

from src.utils.utils import copy_files, create_directory, read_yaml

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


def get_data(path_to_config):
    config = read_yaml(path_to_config)
    remote_data_source = config["data"]["source_path"]
    local_data_source = config["data"]["local_path"]

    # Download Data from source_path to local_path
    for src_dir, target_dir in tqdm(
        zip(remote_data_source, local_data_source), total=2, colour = "green"
    ):
        create_directory([target_dir])
        copy_files(src_dir, target_dir)


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config", "-c", default="config/config.yaml")
    parsed_args = args.parse_args()

    try:
        logging.info(" >>>>>> Starting Stage 01")
        get_data(parsed_args.config)
        logging.info("Stage 01 completed successfully >>>>>>")
    except Exception as e:
        logging.exception(e)
        raise e
