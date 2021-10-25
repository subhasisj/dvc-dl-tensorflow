import argparse
import io
import logging
import os

import pandas as pd
from tqdm import tqdm

from src.utils.models import get_vgg16, prepare_model
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


def prepare_base_model(path_to_config, path_to_params):
    config = read_yaml(path_to_config)
    params = read_yaml(path_to_params)

    artifacts_dir = config["artifacts"]["DIR"]
    base_model_dir = config["artifacts"]["BASE_MODEL_DIR"]

    # Create base model directory
    base_model_dir_path = os.path.join(artifacts_dir, base_model_dir)
    create_directory([base_model_dir_path])

    model_name = config["artifacts"]["BASE_MODEL_NAME"]
    model_path = os.path.join(base_model_dir_path, model_name)

    model = get_vgg16(input_shape=params["IMAGE_SIZE"],model_path=model_path)

    model = prepare_model(
        model,
        classes=params["CLASSES"],
        freeze_till=2,
        learning_rate=params["LEARNING_RATE"],
        freeze_all=False,
    )

    def _log_model_summary(model):
        with io.StringIO() as stream:
            model.summary(print_fn=lambda x: stream.write(f"{x}\n"))
            summary_str = stream.getvalue()
        return summary_str


    logging.info("Model Summary:")
    logging.info(f"{_log_model_summary(model)}")

    updated_base_model_path = os.path.join(
        base_model_dir_path, config["artifacts"]["UPDATED_MODEL_NAME"]
    )

    logging.info(f"Saving model to {updated_base_model_path}")

    model.save(updated_base_model_path)


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config", "-c", default="config/config.yaml")
    args.add_argument("--params", "-p", default="params.yaml")
    parsed_args = args.parse_args()

    try:
        logging.info(" >>>>>> Starting Stage 01 - Preparing Base Model")
        prepare_base_model(parsed_args.config, parsed_args.params)
        logging.info("Stage 02 completed successfully >>>>>>")
    except Exception as e:
        logging.exception(e)
        raise e
