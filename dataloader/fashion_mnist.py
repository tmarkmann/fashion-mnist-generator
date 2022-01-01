import tensorflow as tf
import numpy as np

class FashionMNISTDataset():
    def __init__(self, batch_size=128, buffer_size=2048, augmentation=False):

        (train_images, _), (test_images, _) = tf.keras.datasets.fashion_mnist.load_data()
        train_images = np.reshape(train_images, (-1, 28, 28, 1))
        test_images = np.reshape(test_images, (-1, 28, 28, 1))

        self.ds_combined = self._build_train_pipeline(np.concatenate([train_images, test_images]), batch_size, buffer_size, augmentation)
        self.ds_train = self._build_train_pipeline(train_images, batch_size, buffer_size, augmentation)
        self.ds_test = self._build_test_pipeline(test_images, batch_size)

    def _build_train_pipeline(self, ds_images, batch_size, buffer_size, augmentation):
        ds = tf.data.Dataset.from_tensor_slices(ds_images)
        ds = ds.map(self.preprocess, num_parallel_calls=tf.data.AUTOTUNE)
        ds = ds.shuffle(buffer_size)
        ds = ds.batch(batch_size)
        if augmentation:
            ds = ds.map(self.augment_data, num_parallel_calls=tf.data.AUTOTUNE)
        ds = ds.prefetch(tf.data.AUTOTUNE)
        return ds

    def _build_test_pipeline(self, ds_images, batch_size):
        ds = tf.data.Dataset.from_tensor_slices(ds_images)
        ds = ds.map(self.preprocess, num_parallel_calls=tf.data.AUTOTUNE)
        ds = ds.batch(batch_size)
        ds = ds.prefetch(tf.data.AUTOTUNE)
        return ds

    def preprocess(self, image, height=28, width=28):
        image = tf.image.resize(image, [height, width])
        return tf.cast(image, tf.float32) / 255.
    
    def augment_data(self, image):
        # TODO
        return image

