import json
import logging
import os
import pprint
import shutil
import sys
import time

import yaml
from tqdm import tqdm


def read_yaml(path_to_yaml: str) -> dict:
    with open(path_to_yaml) as yaml_file:
        content = yaml.load(yaml_file, Loader=yaml.FullLoader)
    logging.info('YAML File: {} parsed Successfully'.format(path_to_yaml))
    return content


def create_directory(dir_paths: list):
    for dir_path in dir_paths:
        os.makedirs(dir_path, exist_ok=True)
        logging.info(f"creating directory: {dir_path}")


def save_metrics(metrics: dict, save_path: str) -> None:
    with open(save_path, "w") as out:
        json.dump(metrics, out, indent=4)
    logging.info(f"saving metrics to {save_path}")


def copy_files(src_path: str, dest_path: str):
    list_of_files = [f for f in os.listdir(src_path) ]
    for file in tqdm(
        list_of_files,
        desc=f"copy files from {src_path} to {dest_path}",
        colour="green",
        total=len(list_of_files),
    ):
        src_file_path = os.path.join(src_path, file)
        dest_file_path = os.path.join(dest_path, file)

        shutil.copy(src_file_path, dest_file_path)

def get_timestamp(filename: str) -> str:
    current_time = time.asctime(time.localtime()).replace(" ", "_").replace(":", "_")
    return f"{filename}_{current_time}"


if __name__ == "__main__":
    pp = pprint.PrettyPrinter(indent=4)
    pp.pprint(read_yaml(sys.argv[1]))

    pp.pprint(get_timestamp("test_callback"))
