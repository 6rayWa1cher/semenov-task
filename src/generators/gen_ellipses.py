import csv
import random
import string
from argparse import ArgumentParser
from dataclasses import dataclass, fields, astuple
from pathlib import Path

from PIL import Image, ImageDraw

from common import generate_position_within_borders, logger, elapsed_timer, clear_folder
from tqdm import trange

MAP_FILENAME = 'map.csv'


@dataclass
class Rectangle:
    x: int
    y: int
    width: int
    height: int

    @property
    def size(self):
        return self.width, self.height

    @property
    def xy(self):
        return (self.x, self.y), (self.x + self.width, self.y + self.height)


class ImageObject:
    def render(self, img: Image, draw: ImageDraw):
        raise NotImplemented


@dataclass
class ImageEllipse(ImageObject):
    location: Rectangle
    color: str

    def render(self, img: Image, draw: ImageDraw):
        draw.ellipse(self.location.xy, self.color)


class ImageDescription:
    width: int
    height: int
    image_objects: list[ImageObject]
    bg_color: str

    def __init__(self, width: int, height: int, bg_color='black'):
        self.width = width
        self.height = height
        self.bg_color = bg_color
        self.image_objects = list()

    @property
    def size(self):
        return self.width, self.height

    def render(self):
        img = Image.new('RGBA', self.size, color=self.bg_color)
        draw = ImageDraw.Draw(img)
        for image_object in self.image_objects:
            image_object.render(img, draw)
        return img

    def add_image_object(self, image_object: ImageObject):
        self.image_objects.append(image_object)


def gen_dimension_length(mins, maxs):
    true_min = max(mins)
    true_max = min(maxs)
    assert true_min <= true_max
    return random.randint(true_min, true_min + true_max)


@dataclass
class GenImageWithEllipseParams:
    width: int
    height: int
    bg_color: str
    ellipse_color: str
    ellipse_padding: int
    min_ellipse_width: int
    max_ellipse_width: int
    min_ellipse_height: int
    max_ellipse_height: int


def gen_image_with_ellipse(p: GenImageWithEllipseParams):
    def gen_ellipse():
        w, h = \
            gen_dimension_length([p.min_ellipse_width], [p.max_ellipse_width, p.width - p.ellipse_padding * 2]), \
                gen_dimension_length([p.min_ellipse_height], [p.max_ellipse_height, p.height - p.ellipse_padding * 2])
        x, y = \
            generate_position_within_borders(0, p.width - w, p.ellipse_padding, p.ellipse_padding), \
                generate_position_within_borders(0, p.height - h, p.ellipse_padding, p.ellipse_padding)
        return ImageEllipse(location=Rectangle(x, y, w, h), color=p.ellipse_color)

    description = ImageDescription(p.width, p.height, p.bg_color)
    description.add_image_object(gen_ellipse())

    return description


def generate_random_filename(extension):
    base = ''.join(random.choice(string.ascii_letters) for _ in range(16))
    return base + '.' + extension


def gen_sample(output_dir, p: GenImageWithEllipseParams):
    filename = generate_random_filename('png')
    output_path = Path(output_dir, filename)
    description = gen_image_with_ellipse(p)
    with description.render() as image:
        image.save(output_path)
    return filename, *astuple(description.image_objects[0].location)


def gen_samples(count: int, output_dir, p: GenImageWithEllipseParams):
    data_list = list()
    for _ in trange(count, desc='generating samples'):
        data = gen_sample(output_dir, p)
        data_list.append(data)
    data_list.sort()
    return data_list


def store_to_csv(path, data, header: list[str]):
    logger.info(f'storing data to csv: {path}')
    try:
        with open(path, 'w', newline='') as out:
            csv_writer = csv.writer(out)
            csv_writer.writerow(header)
            for row in data:
                csv_writer.writerow(row)
    except Exception as e:
        logger.error(f"couldn't save map file")
        logger.exception(e)
        raise


def class_from_args(class_name, arg_dict):
    field_set = {f.name for f in fields(class_name) if f.init}
    filtered_arg_dict = {k: v for k, v in arg_dict.items() if k in field_set}
    return class_name(**filtered_arg_dict)


def process(params):
    params_dict = dict(vars(params))
    logger.info(f'starting, params: {params_dict}')
    with elapsed_timer() as elapsed:
        params.output_dir.mkdir(parents=True, exist_ok=True)
        if params.remove_old:
            clear_folder(params.output_dir)
        data_set = gen_samples(params.count, params.output_dir, class_from_args(GenImageWithEllipseParams, params_dict))
        store_to_csv(Path(params.output_dir, MAP_FILENAME), data_set, ['filename', 'x', 'y', 'w', 'h'])
    logger.info('finished at %.2f seconds' % elapsed())


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-o', dest='output_dir', type=Path, required=True)
    parser.add_argument('-n', dest='count', type=int, required=True)
    parser.add_argument('-r', dest='remove_old', default=False, action='store_true')
    parser.add_argument('--width', dest='width', type=int, default=256)
    parser.add_argument('--height', dest='height', type=int, default=256)
    parser.add_argument('--bg_color', dest='bg_color', type=str, default='black')
    parser.add_argument('--ellipse_color', dest='ellipse_color', type=str, default='white')
    parser.add_argument('--ellipse_padding', dest='ellipse_padding', type=int, default=2)
    parser.add_argument('--min_ellipse_width', dest='min_ellipse_width', type=int, default=16)
    parser.add_argument('--max_ellipse_width', dest='max_ellipse_width', type=int, default=64)
    parser.add_argument('--min_ellipse_height', dest='min_ellipse_height', type=int, default=16)
    parser.add_argument('--max_ellipse_height', dest='max_ellipse_height', type=int, default=64)

    args = parser.parse_args()
    process(args)
