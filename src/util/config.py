from os import environ
from pathlib import Path
from sys import maxsize


class Config:
    root_folder = Path(__file__).parent.parent.parent
    data_folder = root_folder.joinpath("data")
    model_folder = root_folder.joinpath("model")
    max_epochs = maxsize


enable_gpu = True
if not enable_gpu:
    environ['CUDA_VISIBLE_DEVICES'] = '-1'