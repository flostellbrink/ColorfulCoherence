from pathlib import Path
from os import environ

from src.runner import train_and_test
from src.util.config import Config

if __name__ == "__main__":
    # Setup environment to work on server

    # Limit gpu usage
    environ['CUDA_VISIBLE_DEVICES'] = '2'

    # Adjust paths
    Config.data_folder = Path("/scratch/lt2316-h18-resources/coco")
    Config.model_folder = Path("/scratch/gusstefl/model")
    Config.log_folder = Path("/scratch/gusstefl/logs")
    Config.batch_size = 32

    # Run default training
    train_and_test()