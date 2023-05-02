from pathlib import Path
from logzero import logger


def get_root_path():
    curr_path = Path(__file__).parent
    while curr_path.parent.exists() and not curr_path.joinpath('src').is_dir():
        curr_path = curr_path.parent
    return curr_path


def get_resource(name):
    return Path(get_root_path(), 'resources', name)


def get_dataset(name):
    return Path(get_root_path(), 'dataset', name)


def get_model_weights(name):
    return Path(get_root_path(), "model", name)


def clear_folder(main_path):
    for item in main_path.iterdir():
        try:
            item.unlink()
        except OSError as e:
            logger.warn(f"couldn't remove {item}")
            logger.exception(e)


if __name__ == '__main__':
    print(get_root_path())
