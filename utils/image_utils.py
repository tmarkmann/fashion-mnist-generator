import tensorflow as tf
from datetime import datetime


def float_image_to_uint8(image):
    image = (image * 128.0) + 128.0
    return tf.cast(image, tf.uint8)


def get_write2disk_ops(images, n_images, out_dir):
    image_write_ops = []
    for i in range(n_images):
        uint8_image = float_image_to_uint8(images[i, ...])
        write_op = tf.io.write_file(
            '%s/%s' % (out_dir, f'{i}.png'),
            tf.image.encode_png(uint8_image))
        image_write_ops.append(write_op)
    return image_write_ops
