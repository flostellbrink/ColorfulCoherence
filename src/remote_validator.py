from pathlib import Path
from os import environ

from src.util.config import Config
from src.validator import validate

if __name__ == "__main__":
    # Setup environment to work on server

    # Limit gpu usage
    if Config.enable_gpu:
        environ['CUDA_VISIBLE_DEVICES'] = '2'

    # Adjust paths
    Config.data_folder = Path("/scratch/lt2316-h18-resources/coco")
    Config.model_folder = Path("/scratch/gusstefl/model")
    Config.log_folder = Path("/scratch/gusstefl/logs")
    Config.batch_size = 6

    # Run default training
    validate("model/model.h5")