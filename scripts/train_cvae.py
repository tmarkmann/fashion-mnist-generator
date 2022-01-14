from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import app
from absl import flags

import tensorflow as tf
import tensorflow_datasets as tfds
from dataloader.fashion_mnist import FashionMNISTDataset
from  models.cvae import CVAE, HParams

flags.DEFINE_integer('batch_size', 32, 'The number of images in each batch.')

flags.DEFINE_string('log_dir_root', '/tmp/cvae_logdir/',
                    'Root Directory where to write event logs.')

flags.DEFINE_string('checkpoint_dir', '/tmp/cvae_logdir/checkpoint',
                    'Directory to save checkpoints')

flags.DEFINE_integer('epochs', 10,
                     'The maximum number of gradient steps.')

flags.DEFINE_integer('latent_dim', 2, 
                     'Dimensions of the generator noise vector.')

flags.DEFINE_float('lr', 1e-4,
                   'Learning Rate')

flags.DEFINE_integer('grid_size', 5, 'Grid size for image visualization.')

FLAGS = flags.FLAGS


def main(_):
  hparams = HParams(
    FLAGS.batch_size, 
    FLAGS.log_dir_root,
    FLAGS.checkpoint_dir,
    FLAGS.epochs,
    FLAGS.latent_dim,
    FLAGS.lr,
    FLAGS.grid_size)

  dataset = FashionMNISTDataset(batch_size=hparams.batch_size, model_type='vae')

  # get test_sample
  n_images_to_generate = hparams.grid_size ** 2
  for test_batch in dataset.ds_test.take(1):
    test_sample = test_batch['images'][0:n_images_to_generate, :, :, :]
    break

  model = CVAE(hparams)
  model.train(dataset.ds_train, dataset.ds_test, test_sample)
  model.save_weights()
  

if __name__ == '__main__':
    print(f'Eager Execution: {tf.executing_eagerly()}')
    app.run(main)
