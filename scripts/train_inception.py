from datetime import datetime
from pathlib import Path
import tensorflow as tf
import tensorflow_datasets as tfds
from configs.inception_fm_net_config import inception_fm_net_config
from dataloader.fashion_mnist_for_inception import FashionMNISTInceptionDataset
from models.inception_fashion_mnist import FashionMNISTInception
import matplotlib.pyplot as plt

from utils.metrics import mnist_score

config = inception_fm_net_config

dataset = FashionMNISTInceptionDataset(batch_size=128)

print("Test examples: ", len(dataset.ds_test) * 128)
print("Validation examples: ", len(dataset.ds_val) * 128)


score = mnist_score(dataset.ds_test)

print(score)
quit()
OUTPUT_DIR = 'output'
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

model = FashionMNISTInception(config)

optimizer = tf.keras.optimizers.Adam(config["train"]["learn_rate"])
loss = tf.keras.losses.CategoricalCrossentropy(from_logits=False)

metric_auc = tf.keras.metrics.AUC(curve='ROC', multi_label=True, num_labels=10, from_logits=False)
# metric_bin_accuracy = tf.keras.metrics.BinaryAccuracy()

model.compile(
    optimizer=optimizer,
    loss=loss,
    metrics=[metric_auc],
)

# Tensorboard Callback and config logging
log_dir = f'logs/inception' + datetime.now().strftime("%Y-%m-%d--%H.%M")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

config_matrix = [[k, str(w)] for k, w in config["train"].items()]
file_writer = tf.summary.create_file_writer(log_dir)
with file_writer.as_default():
    tf.summary.text("config", tf.convert_to_tensor(config_matrix), step=0)

# Checkpoint Callback to only save best checkpoint
checkpoint_filepath = f'checkpoints/inception' + datetime.now().strftime("%Y-%m-%d--%H.%M") + '/cp.ckpt'
checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=True,
    monitor=metric_auc.name,
    mode='max',
    save_best_only=True)

# Early Stopping if loss plateaus
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    min_delta=0,
    patience=config['train']['early_stopping_patience'])

# Dynamic Learning Rate
dyn_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor="val_loss",
    factor=config['train']['factor_learning_rate'],
    patience=config['train']['patience_learning_rate'],
    mode="min",
    min_lr=config['train']['min_learning_rate'],
)

# Model Training
model.fit(
    dataset.ds_train,
    epochs=config["train"]["epochs"],
    validation_data=dataset.ds_val,
    callbacks=[tensorboard_callback, checkpoint_callback, early_stopping, dyn_lr]
)

# Model Test
model.load_weights(checkpoint_filepath)  # best
result = model.evaluate(
    dataset.ds_test,
    batch_size=config['test']['batch_size'],
    callbacks=[tensorboard_callback])

result = dict(zip(model.metrics_names, result))
print("Evaluation Result: ", result)
result_matrix = [[k, str(w)] for k, w in result.items()]
with file_writer.as_default():
    tf.summary.text("evaluation", tf.convert_to_tensor(result_matrix), step=0)
