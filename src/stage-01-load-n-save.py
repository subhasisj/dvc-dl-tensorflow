from src.utils.utils import read_yaml, create_directory, copy_files
import argparse
import pandas as pd
import os
from tqdm import tqdm


def get_data(path_to_config):
    config = read_yaml(path_to_config)
    remote_data_source = config["data"]["source_path"]
    local_data_source = config["data"]["local_path"]

    # Download Data from source_path to local_path
    for src_dir, target_dir in tqdm(
        zip(remote_data_source, local_data_source), total=2, colour="blue"
    ):
        create_directory([target_dir])
        copy_files(src_dir, target_dir)

    # # save in local storage
    # artifact_dir = os.path.join(config["artifacts"]["storage_dir"],config["artifacts"]["raw_local_dir"])
    # create_directory(dir_paths=[artifact_dir])

    # artifact_file = os.path.join(artifact_dir,config["artifacts"]["raw_local_file"])
    # df.to_csv(artifact_file,sep=",",index=False)


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config", "-c", default="config/config.yaml")
    parsed_args = args.parse_args()

    config = get_data(parsed_args.config)