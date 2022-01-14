import tensorflow as tf
import tensorflow_datasets as tfds

class FashionMNISTDataset():
    def __init__(self, split=['train', 'test', 'test+train'], batch_size=128, buffer_size=2048, model_type='gan'):

        #(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.fashion_mnist.load_data()
        ds_train, ds_test, ds_combined = tfds.load('fashion_mnist:3.0.1', split=split, shuffle_files=True)
        self.model_type = model_type

        self.ds_combined = self._build_train_pipeline(ds_combined, batch_size, buffer_size)
        self.ds_train = self._build_train_pipeline(ds_train, batch_size, buffer_size)
        self.ds_test = self._build_test_pipeline(ds_test, batch_size)

    def _build_train_pipeline(self, ds, batch_size, buffer_size):
        if self.model_type == 'gan':
            ds = ds.map(self.preprocess_gan, num_parallel_calls=tf.data.AUTOTUNE)
        else:
            ds = ds.map(self.preprocess_vae, num_parallel_calls=tf.data.AUTOTUNE)
        ds = ds.shuffle(buffer_size)
        ds = ds.batch(batch_size, drop_remainder=True)
        ds = ds.prefetch(tf.data.AUTOTUNE)
        return ds

    def _build_test_pipeline(self, ds, batch_size):
        if self.model_type == 'gan':
            ds = ds.map(self.preprocess_gan, num_parallel_calls=tf.data.AUTOTUNE)
        else:
            ds = ds.map(self.preprocess_vae, num_parallel_calls=tf.data.AUTOTUNE)
        ds = ds.batch(batch_size)
        ds = ds.prefetch(tf.data.AUTOTUNE)
        return ds

    def preprocess_gan(self, element):
        """Map elements to the example dicts expected by the model."""
        images = (tf.cast(element['image'], tf.float32) - 127.5) / 127.5
        num_classes = 10
        one_hot_labels = tf.one_hot(element['label'], num_classes)
        return {'images': images, 'labels': one_hot_labels}

    def preprocess_vae(self, element):
        """Map elements to the example dicts expected by the model."""
        images = (tf.cast(element['image'], tf.float32)) / 255

        return {'images': images, 'labels': element['label']}
    
    def provide_data(self, split='combined'):
        if split == 'combined':
            ds = self.ds_combined
        elif split == 'train':
            ds = self.ds_train
        else:
            ds = self.ds_test

        next_batch = tf.compat.v1.data.make_one_shot_iterator(ds).get_next()
        images, labels = next_batch['images'], next_batch['labels']
        return images, labels
