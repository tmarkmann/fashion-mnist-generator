from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import app
from absl import flags

import tensorflow as tf
import numpy as np
from  models.cvae import CVAE, HParams

flags.DEFINE_string('checkpoint_dir', '/tmp/cvae_logdir/checkpoint',
                    'Directory to save checkpoints')

flags.DEFINE_string('output_dir', '/tmp/cvae_output',
                    'Directory to save generated images')

flags.DEFINE_integer('n_images', 100,
                     'Number of images to generate.')

flags.DEFINE_integer('latent_dim', 2,
                     'Dimensions of the generator noise vector.')

FLAGS = flags.FLAGS


def main(_):
  hparams = HParams(
    32,
    '/tmp',
    FLAGS.checkpoint_dir,
    1,
    FLAGS.latent_dim,
    0.001,
    1,
  )

  if not tf.io.gfile.exists(FLAGS.output_dir):
    tf.io.gfile.makedirs(FLAGS.output_dir)

  model = CVAE(hparams)
  model.load_weights(FLAGS.checkpoint_dir)
  predictions = model.sample(n_to_generate=FLAGS.n_images)
  for i in range(predictions.shape[0]):
    image = np.expand_dims(predictions[i, :, :, 0], axis=2)
    tf.keras.utils.save_img(f'{FLAGS.output_dir}/{i}.png', image)

if __name__ == '__main__':
    print(f'Eager Execution: {tf.executing_eagerly()}')
    app.run(main)
