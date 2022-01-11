from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow.compat.v1 as tf
import tensorflow_gan as tfgan
import tensorflow_hub as tfhub

MNIST_MODULE = 'https://tfhub.dev/tensorflow/tfgan/eval/mnist/logits/1'

def mnist_score(images, num_batches=1):
  """Get MNIST classifier score.
  Args:
    images: A minibatch tensor of MNIST digits. Shape must be [batch, 28, 28,
      1].
    num_batches: Number of batches to split `generated_images` in to in order to
      efficiently run them through Inception.
  Returns:
    The classifier score, a floating-point scalar.
  """
  images.shape.assert_is_compatible_with([None, 28, 28, 1])
  mnist_classifier_fn = tfhub.load(MNIST_MODULE)
  score = tfgan.eval.classifier_score(images, mnist_classifier_fn, num_batches)
  score.shape.assert_is_compatible_with([])

  return score


def mnist_frechet_distance(real_images, generated_images, num_batches=1):
  """Frechet distance between real and generated images.
  This technique is described in detail in https://arxiv.org/abs/1706.08500.
  Please see TF-GAN for implementation details.
  Args:
    real_images: Real images to use to compute Frechet Inception distance.
    generated_images: Generated images to use to compute Frechet Inception
      distance.
    num_batches: Number of batches to split images into in order to efficiently
      run them through the classifier network.
  Returns:
    The Frechet distance. A floating-point scalar.
  """
  real_images.shape.assert_is_compatible_with([None, 28, 28, 1])
  generated_images.shape.assert_is_compatible_with([None, 28, 28, 1])
  mnist_classifier_fn = tfhub.load(MNIST_MODULE)
  frechet_distance = tfgan.eval.frechet_classifier_distance(
      real_images, generated_images, mnist_classifier_fn, num_batches)
  frechet_distance.shape.assert_is_compatible_with([])

  return frechet_distance