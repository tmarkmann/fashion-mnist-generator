import tensorflow as tf

class GAN(tf.keras.Model):
    """ a basic GAN class 
    Extends:
        tf.keras.Model
    """

    def __init__(self, generator=None, discriminator=None, latent_dim=128):
        super(GAN, self).__init__()
        self.latent_dim = latent_dim
        if generator:
            self.gen = generator
        else:
            self.gen = self._build_generator()
        if discriminator:
            self.disc = discriminator
        else:
            self.disc = self._build_discriminator()
        
    def compile(self, d_optimizer, g_optimizer, loss_fn):
        super(GAN, self).compile()
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
        self.loss_fn = loss_fn

    def _build_generator(self):
        return tf.keras.Sequential(
            [
                tf.keras.Input(shape=(self.latent_dim,)),
                tf.keras.layers.Dense(7 * 7 * 128),
                tf.keras.layers.LeakyReLU(alpha=0.2),
                tf.keras.layers.Reshape((7, 7, 128)),
                tf.keras.layers.Conv2DTranspose(128, (4, 4), strides=(2, 2), padding="same"),
                tf.keras.layers.LeakyReLU(alpha=0.2),
                tf.keras.layers.Conv2DTranspose(128, (4, 4), strides=(2, 2), padding="same"),
                tf.keras.layers.LeakyReLU(alpha=0.2),
                tf.keras.layers.Conv2D(1, (7, 7), padding="same", activation="sigmoid"),
            ],
            name="generator",
        )

    def _build_discriminator(self):
        return tf.keras.Sequential(
            [
                tf.keras.Input(shape=(28, 28, 1)),
                tf.keras.layers.Conv2D(64, (3, 3), strides=(2, 2), padding="same"),
                tf.keras.layers.LeakyReLU(alpha=0.2),
                tf.keras.layers.Conv2D(128, (3, 3), strides=(2, 2), padding="same"),
                tf.keras.layers.LeakyReLU(alpha=0.2),
                tf.keras.layers.GlobalMaxPooling2D(),
                tf.keras.layers.Dense(1),
            ],
            name="discriminator",
        )

    def train_step(self, real_images):
        if isinstance(real_images, tuple):
            real_images = real_images[0]
        # Sample random points in the latent space
        batch_size = tf.shape(real_images)[0]
        random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim))

        # Decode them to fake images
        generated_images = self.gen(random_latent_vectors)

        # Combine them with real images
        combined_images = tf.concat([generated_images, real_images], axis=0)

        # Assemble labels discriminating real from fake images
        labels = tf.concat(
            [tf.ones((batch_size, 1)), tf.zeros((batch_size, 1))], axis=0
        )
        # Add random noise to the labels - important trick!
        labels += 0.05 * tf.random.uniform(tf.shape(labels))

        # Train the discriminator
        with tf.GradientTape() as tape:
            predictions = self.disc(combined_images)
            d_loss = self.loss_fn(labels, predictions)
        grads = tape.gradient(d_loss, self.disc.trainable_weights)
        self.d_optimizer.apply_gradients(
            zip(grads, self.disc.trainable_weights)
        )

        # Sample random points in the latent space
        random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim))

        # Assemble labels that say "all real images"
        misleading_labels = tf.zeros((batch_size, 1))

        # Train the generator (note that we should *not* update the weights
        # of the discriminator)!
        with tf.GradientTape() as tape:
            predictions = self.disc(self.gen(random_latent_vectors))
            g_loss = self.loss_fn(misleading_labels, predictions)
        grads = tape.gradient(g_loss, self.gen.trainable_weights)
        self.g_optimizer.apply_gradients(zip(grads, self.gen.trainable_weights))
        return {"d_loss": d_loss, "g_loss": g_loss}

    def generate(self, z):
        return self.gen(z)

    def discriminate(self, x):
        return self.disc(x)
