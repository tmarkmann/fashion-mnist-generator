import tensorflow as tf
import numpy as np

OUTDIR = 'exp_images/real'

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()

if not tf.io.gfile.exists(OUTDIR):
    tf.io.gfile.makedirs(OUTDIR)

for i, elem in enumerate(x_test[:1000]):
    tf.keras.utils.save_img(f'{OUTDIR}/{i}.png', np.expand_dims(elem, axis=2))