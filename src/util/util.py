from pathlib import Path
from keras import Model
from keras.callbacks import TensorBoard, ModelCheckpoint
from src.util.config import Config


class Util:
    def __init__(self, network: str):
        self.network = network
        folder = Config.log_folder.joinpath(network)
        self.latest_folder = sorted(list(map(lambda f: int(f.name), folder.glob("*"))) + [-1])[-1]

    def tensor_board(self) -> TensorBoard:
        """
        Create a tensor board instance in a new sub folder
        """
        folder = Config.log_folder.joinpath(self.network).joinpath(str(self.latest_folder + 1))
        folder.mkdir(parents=True, exist_ok=True)
        return TensorBoard(log_dir=str(folder))

    def model_checkpoint(self) -> ModelCheckpoint:
        """
        Create checkpoints in new sub folder
        :return:
        """
        folder = Config.model_folder.joinpath(self.network).joinpath(str(self.latest_folder + 1))
        folder.mkdir(parents=True, exist_ok=True)
        file = folder.joinpath("checkpoint-{epoch:02d}-l:{loss:.2f}-vl:{val_loss:.2f}.h5")
        return ModelCheckpoint(str(file), period=10)

    def save_model(self, model: Model):
        folder = Config.model_folder.joinpath(self.network).joinpath(str(self.latest_folder + 1))
        folder.mkdir(parents=True, exist_ok=True)
        file = folder.joinpath("model.h5")
        model.save(str(file))


def latest_checkpoint(network: str) -> Path:
    folder = Config.model_folder.joinpath(network)
    latest_folder = sorted(folder.glob("*"), key=lambda f: int(f.name))[-1]
    return sorted(latest_folder.glob("*"), key=lambda file: file.stat().st_mtime)[-1]
