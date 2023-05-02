import random

import tensorflow as tf
import pandas as pd
from pathlib import Path
import numpy as np

from PIL import Image, ImageDraw
from keras.optimizers import Adam

from common.path import get_dataset, get_model_weights

LIMIT_GPU_MEMORY = False
LIMIT_GPU_MEMORY_TO = 1024 * 6
RANDOM_SEED = 42
IMAGE_WIDTH_EXTERNAL = 256
IMAGE_HEIGHT_EXTERNAL = 256
IMAGE_WIDTH_INTERNAL = 64
IMAGE_HEIGHT_INTERNAL = 64
INTERPOLATION = 'nearest'


def normalize_labels(labels):
    # x, y, w, h
    labels = labels.copy().astype('float32')
    # labels[:, 2:4] += labels[:, 0:2]
    # x1, y1, x2, y2
    labels[:, (0, 2)] /= IMAGE_WIDTH_EXTERNAL
    labels[:, (1, 3)] /= IMAGE_HEIGHT_EXTERNAL
    return labels


def prediction_to_xy(predict):
    predict = predict.copy()
    predict[:, (0, 2)] *= IMAGE_WIDTH_EXTERNAL
    predict[:, (1, 3)] *= IMAGE_HEIGHT_EXTERNAL
    predict[:, 2:4] += predict[:, 0:2]
    predict = predict.round().astype('int')
    x1, y1, x2, y2 = predict[0]
    print(x1, y1, x2, y2)
    return min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)


def draw_prediction(image, prediction):
    xy = prediction_to_xy(prediction)
    draw = ImageDraw.Draw(image)
    draw.rectangle(xy, outline='red')
    return image


def np2d_to_list_of_tuples(a):
    return list(tuple(a[i]) for i in range(len(a)))


def create_dataset_from_map(data_map):
    data_map = data_map.sort_values('filename')
    labels = data_map[['x', 'y', 'w', 'h']].to_numpy()
    labels = normalize_labels(labels)
    labels_list = np2d_to_list_of_tuples(labels)
    training, validation = tf.keras.utils.image_dataset_from_directory(
        get_dataset('gen_ellipses'),
        labels=labels_list,
        color_mode='grayscale',
        image_size=(IMAGE_WIDTH_INTERNAL, IMAGE_HEIGHT_INTERNAL), interpolation=INTERPOLATION,
        shuffle=True, seed=RANDOM_SEED,
        batch_size=32,
        validation_split=0.1, subset="both"
    )
    return training, validation


def load_dataset(data_dir):
    print('preparing dataset')
    data_map = pd.read_csv(Path(data_dir, 'map.csv'), sep=',', header=0)
    training, validation = create_dataset_from_map(data_map)
    print('prepared dataset')
    return training, validation


def load_single_file_dataset(file):
    image_for_arr = tf.keras.utils.load_img(
        file, color_mode='grayscale',
        target_size=(IMAGE_WIDTH_INTERNAL, IMAGE_HEIGHT_INTERNAL), interpolation=INTERPOLATION
    )
    image_for_pil = Image.open(file)
    arr = tf.keras.utils.img_to_array(image_for_arr, dtype='float32')
    arr = arr.reshape((1, IMAGE_WIDTH_INTERNAL, IMAGE_HEIGHT_INTERNAL, 1))
    return arr, image_for_pil


def setup_environment():
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        if LIMIT_GPU_MEMORY:
            try:
                tf.config.set_logical_device_configuration(
                    gpus[0],
                    [tf.config.LogicalDeviceConfiguration(memory_limit=LIMIT_GPU_MEMORY_TO)])
                logical_gpus = tf.config.list_logical_devices('GPU')
                print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
            except RuntimeError as e:
                # Virtual devices must be set before GPUs have been initialized
                print(e)
        else:
            tf.config.experimental.set_memory_growth(gpus[0], True)
    tf.random.set_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    random.seed(RANDOM_SEED)


def get_model(print_summary=True):
    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=(IMAGE_WIDTH_INTERNAL, IMAGE_HEIGHT_INTERNAL, 1)),
        tf.keras.layers.Flatten(),
        # tf.keras.layers.Dense(5000, activation=tf.nn.relu),
        # tf.keras.layers.Dense(500, activation=tf.nn.sigmoid),
        tf.keras.layers.Dense(200, activation=tf.nn.sigmoid),
        tf.keras.layers.Dense(4, activation=tf.nn.sigmoid)
    ])

    model.compile(optimizer=Adam(learning_rate=0.00065), loss='mse', metrics=['accuracy'])

    if print_summary:
        model.summary()

    return model


def try_load():
    path = get_model_weights('detect_ellipse/model')
    if path.exists():
        return tf.keras.models.load_model(path)
    else:
        return None


def save_model(model):
    path = get_model_weights('detect_ellipse/model')
    model.save(path)


def get_trained_model():
    model = try_load()
    if model is None:
        print("couldn't load model, staring train procedure")
        model = get_model()
        training, validation = load_dataset(get_dataset('gen_ellipses'))

        model.fit(training, epochs=20, validation_data=validation, verbose=1)
        print("finished training")
        save_model(model)
    else:
        print("loaded weights, skipped train procedure")
        model.summary()
    return model


if __name__ == '__main__':
    model = get_trained_model()
    # aDVGWyTqYhYMnxOi.png, ACCPNEaSatKFPAeU.png, AZMVgEShWvAToAkQ.png
    test, image = load_single_file_dataset(get_dataset('gen_ellipses/ACCPNEaSatKFPAeU.png'))
    prediction = model.predict(test)
    image = draw_prediction(image, prediction)
    # image.save("a.png")
    image.show()
