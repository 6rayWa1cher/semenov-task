import csv
import random
import string
from argparse import ArgumentParser
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
from logzero import logger

from common import generate_position_within_borders
from common.path import clear_folder

DEFAULT_IMAGE_FILE_EXT = 'png'
DEFAULT_MAP_FILE_FILENAME = 'map.csv'
BACKGROUND_FOLDER_LOCATION = 'resources/gen_captcha_images/bg'
DIGIT_CELL_WIDTH = 50
DIGIT_CELL_HEIGHT = 50
DIGIT_CELL_TOP_PADDING = 20
DIGIT_CELL_BOTTOM_PADDING = 20
DIGIT_CELL_LEFT_PADDING = 20
DIGIT_CELL_RIGHT_PADDING = 20
DIGIT_CELL_FONT_NAME = 'arial.ttf'
DIGIT_CELL_FONT_SIZE = 15
DIGIT_CELL_FONT_COLOR = 'white'
NUMBER_DIGITS = 4
ANGLE_MAX_DEVIATION = 30
DIGIT_CELL_COLOR_PER_BG = {
    'bg1.png': 'white',
    'bg2.png': 'white',
    'bg3.png': 'black',
    'bg4.png': 'black',
    'bg5.png': 'white',
    'bg6.png': 'white',
    'bg7.png': 'black',
    'bg8.png': 'black'
}


@dataclass
class Background:
    image: Image
    contract_color: str


def load_image_from_fs(path):
    with Image.open(path) as img:
        img.load()
    return img


def generate_number_with_n_digits(n):
    return random.randrange(10 ** (n - 1), 10 ** n)


def preprocess_backgrounds(path):
    backgrounds = list()
    for p in path.iterdir():
        if not p.is_file() or not p.name.endswith(DEFAULT_IMAGE_FILE_EXT):
            logger.debug(f'skipping background {p}')
            continue

        bg = load_image_from_fs(p)

        assert bg.width == 512, f'background {p} has width {bg.width}, not 512'
        assert bg.height == 512, f'background {p} has height {bg.height}, not 512'

        backgrounds.append(Background(image=bg, contract_color=DIGIT_CELL_COLOR_PER_BG[p.name]))
    return backgrounds


def get_number_digit_count(number):
    return len(str(number))


def generate_angle(max_deviation=ANGLE_MAX_DEVIATION):
    return random.randint(-max_deviation, max_deviation)


def generate_image_with_number(number, cell_width=DIGIT_CELL_WIDTH, cell_height=DIGIT_CELL_HEIGHT,
                               font_name=DIGIT_CELL_FONT_NAME, font_size=DIGIT_CELL_FONT_SIZE,
                               font_color=DIGIT_CELL_FONT_COLOR,
                               top_padding=DIGIT_CELL_TOP_PADDING, bottom_padding=DIGIT_CELL_BOTTOM_PADDING,
                               left_padding=DIGIT_CELL_LEFT_PADDING, right_padding=DIGIT_CELL_RIGHT_PADDING):
    digit_count = get_number_digit_count(number)
    img_width, img_height = cell_width * digit_count, cell_height
    img = Image.new('RGBA', (img_width, img_height))
    font = ImageFont.truetype(font_name, font_size, encoding="unic")
    for i, digit in enumerate(str(number)):
        cell_x, cell_y = cell_width * i, 0
        with Image.new('RGBA', (cell_width, cell_height)) as digit_img:
            digit_x, digit_y = \
                generate_position_within_borders(0, cell_width, left_padding, right_padding), \
                generate_position_within_borders(0, cell_height, top_padding, bottom_padding)
            digit_draw = ImageDraw.Draw(digit_img)
            digit_draw.text((digit_x, digit_y), digit, fill=font_color, font=font)
            img.paste(digit_img.rotate(generate_angle(), resample=Image.Resampling.BICUBIC), (cell_x, cell_y))
    return img


def place_image_onto_background(image, bg):
    new_bg = bg.copy()
    x, y = \
        generate_position_within_borders(0, new_bg.width, 0, image.width), \
        generate_position_within_borders(0, new_bg.height, 0, image.height)
    new_bg.paste(image, (x, y), image)
    return new_bg


def create_sample_image(backgrounds, digits=NUMBER_DIGITS):
    number = generate_number_with_n_digits(digits)
    bg = random.choice(backgrounds)
    number_img = generate_image_with_number(number, font_color=bg.contract_color)
    image = place_image_onto_background(number_img, bg.image)
    return number, image


def generate_random_filename(extension):
    base = ''.join(random.choice(string.ascii_letters) for _ in range(16))
    return base + '.' + extension


def generate_images(backgrounds, count):
    image = None
    for i in range(count):
        try:
            number, image = create_sample_image(backgrounds)
            random_file_name = generate_random_filename(DEFAULT_IMAGE_FILE_EXT)

            yield image, number, random_file_name

            logger.info(f'generated {random_file_name}, {i + 1}/{count}')
        except Exception as e:
            logger.error("couldn't generate an image")
            logger.exception(e)
            raise
        finally:
            if image:
                image.close()


def process(output, count, remove_old):
    try:
        backgrounds = preprocess_backgrounds(Path(BACKGROUND_FOLDER_LOCATION))
    except OSError as e:
        logger.error("couldn't prepare backgrounds")
        logger.exception(e)
        raise

    logger.info(f'starting, output={output}, count={count}')

    main_path = Path(output) if output else None
    if main_path:
        try:
            main_path.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            logger.error(f"couldn't create a directory with the path {main_path}")
            logger.exception(e)
            raise

        if remove_old:
            clear_folder(main_path)

        filename_to_number = dict()
        for image, number, random_file_name in generate_images(backgrounds, count):
            image.save(Path(main_path, random_file_name))
            filename_to_number[random_file_name] = number

        try:
            with open(Path(main_path, DEFAULT_MAP_FILE_FILENAME), 'w', newline='') as out:
                csv_writer = csv.writer(out)
                csv_writer.writerow(['filename', 'number'])
                for filename, number in filename_to_number.items():
                    csv_writer.writerow([filename, number])
        except Exception as e:
            logger.error(f"couldn't save map file")
            logger.exception(e)
            raise
    else:
        for image, _, _ in generate_images(backgrounds, count):
            image.show()

    for bg in backgrounds:
        bg.image.close()

    logger.info(f'finished')


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-o', dest='output', type=Path)
    parser.add_argument('-n', dest='count', type=int, required=True)
    parser.add_argument('-r', dest='remove_old', default=False, action='store_true')
    args = parser.parse_args()
    process(**vars(args))
