import unittest
import numpy as np
import tensorflow as tf
from numpy.testing import assert_almost_equal

from tensorflow.python.keras import layers, activations, optimizers, losses
from tensorflow.python.keras.engine.input_layer import InputLayer
from tensorflow.python.keras.models import Sequential


class EnvTest(unittest.TestCase):
    def test_tensorflow_computing(self):
        x = np.array([[0, 0],
                      [1, 1],
                      [1, 0],
                      [0, 1]], dtype=np.float32)

        y = np.array([[0],
                      [0],
                      [1],
                      [1]], dtype=np.float32)

        model = Sequential()
        model.add(InputLayer(input_shape=(2,)))
        model.add(layers.Dense(10, activation=activations.sigmoid))
        model.add(layers.Dense(1, activation=activations.sigmoid))

        model.compile(optimizer=optimizers.adam_v2.Adam(learning_rate=0.02),
                      loss=losses.MeanSquaredError(),
                      metrics=['mse', 'binary_accuracy'])
        model.summary()
        model.fit(x, y, batch_size=1, epochs=500)
        predictions = model.predict_on_batch(x)
        assert_almost_equal(np.round(predictions), np.array([[0], [0], [1], [1]]))

    def test_tensorflow_gpu_available(self):
        gpus = tf.config.list_physical_devices('GPU')
        print(gpus)
        self.assertGreaterEqual(len(gpus), 1, 'No GPU found')


if __name__ == '__main__':
    unittest.main()
