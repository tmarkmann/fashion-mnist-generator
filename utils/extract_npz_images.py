import tensorflow as tf
import numpy as np
import os

def write_npz_to_images(npz_paths, output_dir):
    if not tf.io.gfile.exists(output_dir):
            tf.io.gfile.makedirs(output_dir)
    counter = 0
    for npz_file in npz_paths:
        images = np.load(npz_file)
        for image in images['samples']:
            image_path = os.path.join(output_dir, f'{counter}.png')
            tf.keras.utils.save_img(image_path, np.transpose(image, axes=[1, 2, 0]))
            counter += 1