import collections
from datetime import datetime

from numpy.lib import utils
from tensorflow_gan.python.eval.classifier_metrics import classifier_score
import tensorflow.compat.v1 as tf
import tensorflow_gan as tfgan
from tensorflow_gan.examples.mnist.networks import unconditional_discriminator, unconditional_generator
from tensorflow_gan.examples import evaluation_helper as evaluation
from dataloader.fashion_mnist import FashionMNISTDataset
from utils.image_utils import float_image_to_uint8, get_write2disk_ops
from utils.metrics import mnist_frechet_distance, mnist_score


HParams = collections.namedtuple('HParams', [
    'model_type',
    'loss_type',
    'batch_size',
    'log_dir_root',
    'max_number_of_steps',
    'grid_size',
    'noise_dims',
    'lr_generator',
    'lr_discriminator',
    'n_eval_images',
])

EvalParams = collections.namedtuple('EvalParams', [
    'checkpoint_dir',
    'n_images',
])

class GAN():
    def __init__(self, hparams):
        self.hparams = hparams

        # log dir
        self.train_log_dir = f'{hparams.log_dir_root}/train_{hparams.model_type}_{hparams.loss_type}_{datetime.now().strftime("%d-%m--%H.%M")}'
        self.gen_out_dir = f'{hparams.log_dir_root}/images_{hparams.model_type}_{hparams.loss_type}_{datetime.now().strftime("%d-%m--%H.%M")}'

        self._build_dataset()
        self.gan_model = self._build_model()
        self.gan_loss = self._build_loss()
        self.gen_optimizer, self.disc_optimizer = self._build_optimizer()

        tfgan.eval.add_gan_model_image_summaries(
            self.gan_model, hparams.grid_size)
        # tfgan.eval.add_regularization_loss_summaries(self.gan_model)

    def _build_dataset(self):
        with tf.name_scope('inputs'), tf.device('/cpu:0'):
            dataset = FashionMNISTDataset(batch_size=self.hparams.batch_size)
            self.ds_images, self.ds_labels = dataset.provide_data()

    def _build_model(self, model_type='tf_gan_unconditional'):
        if model_type == 'tf_gan_unconditional':
            generator = unconditional_generator
            discriminator = unconditional_discriminator
        else:
            # TODO add more networks
            generator = tfgan.examples.mnist.networks.unconditional_generator
            discriminator = tfgan.examples.mnist.networks.unconditional_discriminator

        return tfgan.gan_model(
            generator_fn=generator,
            discriminator_fn=discriminator,
            real_data=self.ds_images,
            generator_inputs=tf.random.normal(
                [self.hparams.batch_size, self.hparams.noise_dims]))

    def _build_loss(self, loss_type='wasserstein'):
        with tf.name_scope('loss'):
            if loss_type == 'wasserstein':
                gen_loss = tfgan.losses.wasserstein_generator_loss
                disc_loss = tfgan.losses.wasserstein_discriminator_loss
            elif loss_type == 'minimax':
                gen_loss = tfgan.losses.minimax_generator_loss
                disc_loss = tfgan.losses.minimax_discriminator_loss
            else:
                # TODO Add loss functions
                gen_loss = tfgan.losses.wasserstein_generator_loss
                disc_loss = tfgan.losses.wasserstein_discriminator_loss

            return tfgan.gan_loss(
                self.gan_model,
                generator_loss_fn=gen_loss,
                discriminator_loss_fn=disc_loss,
                add_summaries=True)

    def _build_optimizer(self, optimizer_type='tfgan_adam'):
        if optimizer_type == 'tfgan_adam':
            gen_optimizer = tf.train.AdamOptimizer(
                self.hparams.lr_generator, 0.5)
            disc_optimizer = tf.train.AdamOptimizer(
                self.hparams.lr_discriminator, 0.5)
            return gen_optimizer, disc_optimizer

    def train(self):
        if not tf.io.gfile.exists(self.train_log_dir):
            tf.io.gfile.makedirs(self.train_log_dir)

        #tf.summary.scalar('MNIST_Classifier_score', mnist_score(self.ds_images))

        with tf.variable_scope('Generator', reuse=tf.compat.v1.AUTO_REUSE):
            images = unconditional_generator(
                tf.random.normal([100, self.hparams.noise_dims]),
                is_training=False)
        frechet = mnist_frechet_distance(self.ds_images, images)
        #classifier_score = mnist_score(images)
        #tf.summary.scalar('MNIST_Frechet_distance', frechet)
        #tf.summary.scalar('MNIST_Classifier_score', classifier_score)

        #with tf.name_scope('train'):
        train_ops = tfgan.gan_train_ops(
            self.gan_model,
            self.gan_loss,
            generator_optimizer=self.gen_optimizer,
            discriminator_optimizer=self.disc_optimizer,
            summarize_gradients=False,
            aggregation_method=tf.AggregationMethod.EXPERIMENTAL_ACCUMULATE_N)

        status_message = tf.strings.join([
            'Starting train step: ',
            tf.as_string(tf.train.get_or_create_global_step())],
            name='status_message',)
        
        tfgan.gan_train(
            train_ops,
            hooks=[
                tf.estimator.StopAtStepHook(
                    num_steps=self.hparams.max_number_of_steps),
                tf.estimator.LoggingTensorHook(
                    [status_message], every_n_iter=10)
            ],
            logdir=self.train_log_dir,
            get_hooks_fn=tfgan.get_joint_train_hooks(),
            save_checkpoint_secs=60)

    def generate_images(self, evalparams):
        if evalparams.checkpoint_dir is not '':
            checkpoint_dir = evalparams.checkpoint_dir
        else:
            checkpoint_dir = self.train_log_dir

        with tf.variable_scope('Generator', reuse=True):
            images = unconditional_generator(
                tf.random.normal(
                    [evalparams.n_images, self.hparams.noise_dims]),
                is_training=False)
        image_write_ops = get_write2disk_ops(images, evalparams.n_images, self.gen_out_dir)
        
        evaluation.evaluate_repeatedly(
            checkpoint_dir,
            hooks=[
                evaluation.StopAfterNEvalsHook(1)
            ],
            eval_ops=image_write_ops,
            max_number_of_evaluations=1)
