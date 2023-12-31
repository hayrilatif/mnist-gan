import tensorflow_datasets as tfds
import tensorflow as tf

from hyperparams import Params

#Returns dataset.
def get_dataset():
    data=tfds.load("mnist")
    train_data=data["train"]
    train_data=train_data.map(lambda x:tf.cast(x["image"],tf.float32)/255.).batch(Params.batchsize).prefetch(tf.data.AUTOTUNE)
    return train_data