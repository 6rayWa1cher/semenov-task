import csv
import random
import string
from argparse import ArgumentParser
from functools import cache
from os import listdir
from pathlib import Path

from PIL import Image
from logzero import logger
from tqdm import trange

from common import get_resource, clear_folder
from common.path import get_dataset


def load_image_from_fs(path):
    with Image.open(path) as img:
        img.load()
    return img


def load_backgrounds():
    path = get_resource('gen_labels_images')
    backgrounds = list()
    for p in path.iterdir():
        if not p.is_file() or not p.name.endswith('.png'):
            logger.debug(f'skipping background {p}')
            continue

        bg = load_image_from_fs(p)

        assert bg.width == 512, f'background {p} has width {bg.width}, not 512'
        assert bg.height == 512, f'background {p} has height {bg.height}, not 512'

        backgrounds.append(bg)
    return backgrounds


@cache
def get_dataset_files():
    labels_path = Path(get_dataset('gen_labels'), 'train', 'img')
    files = listdir(labels_path)
    return labels_path, files


def load_sample_label():
    labels_path, files = get_dataset_files()
    file = random.choice(files)
    path = Path(labels_path, file)
    return load_image_from_fs(path)


def get_random_background(backgrounds):
    return backgrounds[random.randrange(len(backgrounds))]


def generate_size(width, height):
    between_max_and_curr = 256 / width
    scale_factor = random.uniform(0.5 * between_max_and_curr, between_max_and_curr)
    return round(width * scale_factor), round(height * scale_factor)


def generate_position_within_borders(size_of_bg, size_of_img, begin_padding, end_padding):
    assert size_of_bg > size_of_img + begin_padding + end_padding
    return random.randrange(begin_padding, size_of_bg - size_of_img - end_padding)


def create_sample_label(backgrounds):
    background = get_random_background(backgrounds)
    img = background.copy()

    label_img = load_sample_label()

    if label_img.width > 512 // 2:
        target_width, target_height = generate_size(label_img.width, label_img.height)
        label_img = label_img.resize((target_width, target_height))

    x = generate_position_within_borders(512, label_img.width, 10, 10)
    y = generate_position_within_borders(512, label_img.height, 10, 10)

    img.paste(label_img, (x, y))

    return img, (x, y, label_img.width, label_img.height)


def save_image(output, image, filename):
    image.save(Path(output, filename))


def save_database(output, database):
    try:
        with open(Path(output, 'map.csv'), 'w', newline='') as out:
            csv_writer = csv.writer(out)
            csv_writer.writerow(['filename', 'x', 'y', 'w', 'h'])
            for row in database:
                csv_writer.writerow(list(row))
    except Exception as e:
        logger.error(f"couldn't save map file")
        logger.exception(e)
        raise


def gen_image_name(extension='png'):
    base = ''.join(random.choice(string.ascii_letters) for _ in range(16))
    return base + '.' + extension


def prepare_output_folder(output, remove_old):
    try:
        output.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        logger.error(f"couldn't create a directory with the path {output}")
        logger.exception(e)
        raise

    if remove_old:
        clear_folder(output)


def process(output, count, remove_old):
    prepare_output_folder(output, remove_old)
    backgrounds = load_backgrounds()
    database = list()
    for _ in trange(count):
        image, data = create_sample_label(backgrounds)
        filename = gen_image_name()
        save_image(output, image, filename)
        database.append((filename, *data))
    save_database(output, database)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-o', dest='output', type=Path)
    parser.add_argument('-n', dest='count', type=int, required=True)
    parser.add_argument('-r', dest='remove_old', default=False, action='store_true')
    args = parser.parse_args()
    process(**vars(args))
