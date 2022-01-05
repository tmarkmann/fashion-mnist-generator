import tensorflow as tf
import matplotlib.pyplot as plt

class ExampleGeneratorCallback(tf.keras.callbacks.Callback):
    def __init__(self, output_path):
        super(ExampleGeneratorCallback, self).__init__()
        self.output_path = output_path
        self.noise_vector = tf.random.normal(shape=(8, 128))

    def on_epoch_end(self, epoch, logs=None):
        self.generate_examples(epoch)

    def generate_examples(self, epoch):
        examples = self.model.generate(self.noise_vector)
        self.save_examples(examples, epoch)

    def save_examples(self, examples, epoch):
        n_columns = examples.shape[0]
        fig, axs = plt.subplots(ncols=n_columns, nrows=1, figsize=(2 * n_columns, 2))
        for axi in range(n_columns):
            axs[axi].matshow(
                        examples.numpy()[axi].squeeze(), cmap=plt.cm.Greys, vmin=0, vmax=1
                    )
            axs[axi].axis('off')
        plt.savefig(f'{self.output_path}/epoch_{epoch}.png')