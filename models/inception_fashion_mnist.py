import tensorflow as tf


class FashionMNISTInception(tf.keras.Model):
    """FashionMNISTInception net"""

    def __init__(self, config):
        super(FashionMNISTInception, self).__init__(name='FashionMNISTInception')
        self.config = config
        self._input_shape = (
            self.config['data']['image_height'],
            self.config['data']['image_width'],
            self.config['data']['image_channel']
        )

        self.img_input = tf.keras.Input(shape=self._input_shape)
        self.base_model = tf.keras.applications.InceptionV3(
            include_top=False,
            weights="imagenet",
            input_tensor=self.img_input,
            input_shape=self._input_shape,
            pooling='avg',
            classes=10
        )
        self.classifier = tf.keras.layers.Dense(10, activation="softmax", name="predictions")

    def call(self, x):
        x = tf.keras.applications.inception_v3.preprocess_input(x)
        x = self.base_model(x)
        return self.classifier(x)
