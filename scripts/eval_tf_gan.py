from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import app
from absl import flags
from absl import logging

import tensorflow.compat.v1 as tf
from models.tf_gan import GAN, EvalParams, HParams

flags.DEFINE_string('model_type', 'tf_gan_unconditional',
                    'Model type: `tf_gan_unconditional`, tbc')

flags.DEFINE_string('loss_type', 'wasserstein',
                    'Loss Function Type: `wasserstein`, `minimax`, tbc')

flags.DEFINE_integer('batch_size', 32, 'The number of images in each batch.')

flags.DEFINE_string('log_dir_root', '/tmp/tfgan_logdir/',
                    'Root Directory where to write event logs.')

flags.DEFINE_string('checkpoint_dir', '',
                    'Directory where to find model checkpoints.')

flags.DEFINE_integer('max_number_of_steps', 20000,
                     'The maximum number of gradient steps.')

flags.DEFINE_integer('grid_size', 5, 'Grid size for image visualization.')

flags.DEFINE_integer('noise_dims', 64, 
                     'Dimensions of the generator noise vector.')

flags.DEFINE_float('lr_generator', 1e-3,
                   'Generator Learning Rate')

flags.DEFINE_float('lr_discriminator', 1e-4,
                   'Discriminator Learning Rate')

flags.DEFINE_integer('n_eval_images', 500, 
                     'Number of images used for evaluation during training.')

flags.DEFINE_integer('n_images', 1000,
                     'Number of images to be generated for evaluation')

FLAGS = flags.FLAGS


def main(_):
    hparams = HParams(
        FLAGS.model_type,
        FLAGS.loss_type,
        FLAGS.batch_size, 
        FLAGS.log_dir_root,
        FLAGS.max_number_of_steps,
        FLAGS.grid_size,
        FLAGS.noise_dims,
        FLAGS.lr_generator,
        FLAGS.lr_discriminator,
        FLAGS.n_eval_images)
    
    evalparams = EvalParams(
        FLAGS.checkpoint_dir,
        FLAGS.n_images,
    )

    gan = GAN(hparams)
    #gan.train()
    gan.generate_images(evalparams)

if __name__ == '__main__':
    logging.set_verbosity(logging.INFO)
    tf.disable_v2_behavior()
    app.run(main)
