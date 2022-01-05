import tensorflow as tf
import numpy as np
from dataloader.fashion_mnist import FashionMNISTDataset
from models.gan import GAN
from utils.example_generator import ExampleGeneratorCallback
from pathlib import Path

OUTPUT_DIR = 'output'
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

dataset = FashionMNISTDataset(batch_size=128)

gan = GAN(latent_dim=128)
gan.compile(
    d_optimizer=tf.keras.optimizers.Adam(learning_rate=0.0003),
    g_optimizer=tf.keras.optimizers.Adam(learning_rate=0.0003),
    loss_fn=tf.keras.losses.BinaryCrossentropy(from_logits=True),
)

example_generator_cb = ExampleGeneratorCallback(output_path=OUTPUT_DIR)
history = gan.fit(dataset.ds_combined, epochs=20, callbacks=[example_generator_cb])
