import matplotlib.pyplot as plt

from dataloader.fashion_mnist import FashionMNISTDataset
import tensorflow as tf

ds = FashionMNISTDataset()

def show_examples():
    data = ds.ds_train.take(25)
    plt.figure(figsize=(10, 10))
    for index, example in enumerate(data):
        image = example[0].numpy()
        image = tf.image = image.reshape((28,28))
        plt.subplot(5, 5, index+1)
        plt.imshow(image, cmap=plt.cm.binary)
    plt.show()

show_examples()