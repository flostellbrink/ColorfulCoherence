from keras.callbacks import TensorBoard

from src.util.config import Config


def tensor_board(network: str) -> TensorBoard:
    """
    Create a tensor board instance in a new sub folder
    """
    folder = Config.root_folder.joinpath("logs").joinpath(network)
    latest_folder = sorted(list(map(lambda f: int(f.name), folder.glob("*"))) + [-1])[-1]

    new_folder = folder.joinpath(str(latest_folder + 1))
    new_folder.mkdir(parents=True, exist_ok=True)

    return TensorBoard(log_dir=str(new_folder))