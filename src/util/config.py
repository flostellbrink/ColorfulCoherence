from os import environ
from pathlib import Path


class Config:
    root_folder = Path(__file__).parent.parent.parent
    data_folder = root_folder.joinpath("data")
    model_folder = root_folder.joinpath("model")
    log_folder = root_folder.joinpath("logs")
    max_epochs = 100
    batch_size = 1
    epsilon = 1e-10
    enable_gpu = True


if not Config.enable_gpu:
    environ['CUDA_VISIBLE_DEVICES'] = '-1'