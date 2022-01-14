import tensorflow as tf
import numpy as np
import collections
import time
from tensorflow_gan.python.eval import eval_utils 

HParams = collections.namedtuple('HParams', [
    'batch_size',
    'log_dir_root',
    'checkpoint_dir',
    'epochs',
    'latent_dim',
    'lr',
    'grid_size'
])

class CVAE(tf.keras.Model):
  """Convolutional variational autoencoder."""

  def __init__(self, hparams):
    super(CVAE, self).__init__()
    self.hparams = hparams
    if hparams.checkpoint_dir == None:
      hparams.checkpoint_dir = '/tmp/cvae/checkpoint'
    self.latent_dim = hparams.latent_dim
    self.encoder = tf.keras.Sequential(
        [
            tf.keras.layers.InputLayer(input_shape=(28, 28, 1)),
            tf.keras.layers.Conv2D(
                filters=32, kernel_size=3, strides=(2, 2), activation='relu'),
            tf.keras.layers.Conv2D(
                filters=64, kernel_size=3, strides=(2, 2), activation='relu'),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(hparams.latent_dim + hparams.latent_dim),
        ]
    )

    self.decoder = tf.keras.Sequential(
        [
            tf.keras.layers.InputLayer(input_shape=(hparams.latent_dim,)),
            tf.keras.layers.Dense(units=7*7*32, activation=tf.nn.relu),
            tf.keras.layers.Reshape(target_shape=(7, 7, 32)),
            tf.keras.layers.Conv2DTranspose(
                filters=64, kernel_size=3, strides=2, padding='same',
                activation='relu'),
            tf.keras.layers.Conv2DTranspose(
                filters=32, kernel_size=3, strides=2, padding='same',
                activation='relu'),
            tf.keras.layers.Conv2DTranspose(
                filters=1, kernel_size=3, strides=1, padding='same'),
        ]
    )

    self.optimizer = tf.keras.optimizers.Adam(hparams.lr)

  def save_weights(self):
    self.encoder.save_weights(f'{self.hparams.checkpoint_dir}/encoder')
    self.decoder.save_weights(f'{self.hparams.checkpoint_dir}/decoder')

  def load_weights(self, path):
    self.encoder.load_weights(f'{path}/encoder')
    self.decoder.load_weights(f'{path}/decoder')

  @tf.function
  def sample(self, eps=None, n_to_generate=100):
    if eps is None:
      eps = tf.random.normal(shape=(n_to_generate, self.latent_dim))
    return self.decode(eps, apply_sigmoid=True)

  def encode(self, x):
    mean, logvar = tf.split(self.encoder(x), num_or_size_splits=2, axis=1)
    return mean, logvar

  def reparameterize(self, mean, logvar):
    eps = tf.random.normal(shape=mean.shape)
    return eps * tf.exp(logvar * .5) + mean

  def decode(self, z, apply_sigmoid=False):
    logits = self.decoder(z)
    if apply_sigmoid:
      probs = tf.sigmoid(logits)
      return probs
    return logits

  def train(self, ds_train, ds_val, test_sample):
    writer = tf.summary.create_file_writer(f'{self.hparams.log_dir_root}/cvae')
    with writer.as_default():
      assert self.hparams.batch_size >= len(test_sample)
      self.log_test_image_grid(test_sample, 0)

      for epoch in range(1, self.hparams.epochs + 1):
        for train_x in ds_train:
          self.train_step(train_x['images'])

        loss = tf.keras.metrics.Mean()
        for test_x in ds_val:
          loss(self._compute_loss(test_x['images']))
        elbo = -loss.result()
        tf.summary.scalar('elbo', elbo, step=epoch)
        self.log_test_image_grid(test_sample, epoch)
        print(f'Epoch {epoch} finished with elbo={elbo}')

  @tf.function
  def train_step(self, x):
    """Executes one training step and returns the loss.

    This function computes the loss and gradients, and uses the latter to
    update the model's parameters.
    """
    with tf.GradientTape() as tape:
      loss = self._compute_loss(x)
    gradients = tape.gradient(loss, self.trainable_variables)
    self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

  def generate_images(self, test_sample):
    mean, logvar = self.encode(test_sample)
    z = self.reparameterize(mean, logvar)
    return self.sample(z)

  def log_test_image_grid(self, test_sample, epoch):
    images = self.generate_images(test_sample)
    image_shape = images.shape.as_list()[1:3]
    channels = images.shape.as_list()[3]
    image_grid = eval_utils.image_grid(images,
      grid_shape=(self.hparams.grid_size, self.hparams.grid_size),
      image_shape=image_shape,
      num_channels=channels)
    tf.summary.image('test_sample_image', image_grid, step=epoch)

  def _log_normal_pdf(self, sample, mean, logvar, raxis=1):
    log2pi = tf.math.log(2. * np.pi)
    return tf.reduce_sum(
        -.5 * ((sample - mean) ** 2. * tf.exp(-logvar) + logvar + log2pi),
        axis=raxis)

  def _compute_loss(self, x):
    mean, logvar = self.encode(x)
    z = self.reparameterize(mean, logvar)
    x_logit = self.decode(z)
    cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(logits=x_logit, labels=x)
    logpx_z = -tf.reduce_sum(cross_ent, axis=[1, 2, 3])
    logpz = self._log_normal_pdf(z, 0., 0.)
    logqz_x = self._log_normal_pdf(z, mean, logvar)
    return -tf.reduce_mean(logpx_z + logpz - logqz_x)
