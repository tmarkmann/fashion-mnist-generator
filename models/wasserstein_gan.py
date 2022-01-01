import tensorflow_gan as tfgan


def get_wasserstein_loss(model):
    return tfgan.gan_loss(
        model,
        generator_loss_fn=tfgan.losses.wasserstein_generator_loss,
        discriminator_loss_fn=tfgan.losses.wasserstein_discriminator_loss,
        gradient_penalty_weight=1.0)
