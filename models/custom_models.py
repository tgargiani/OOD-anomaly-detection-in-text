from custom_layers import CosFace
from utils import EXTRA_LAYER_ACT_F

import tensorflow as tf
from tensorflow.keras import layers


class CosFaceModel(tf.keras.Model):
    def __init__(self, emb_dim, num_classes, extra_layer=False):
        super(CosFaceModel, self).__init__()
        self.inp = tf.keras.layers.Input(shape=emb_dim)
        self.inp2 = tf.keras.layers.Input(shape=())
        self.cosface = CosFace(num_classes=num_classes)
        self.extra_layer = extra_layer

        if extra_layer:
            self.dense = layers.Dense(emb_dim, EXTRA_LAYER_ACT_F)

    def call(self, inputs, training=None):
        if training:
            x, labels = inputs
        else:
            x = inputs

        if self.extra_layer:
            x = self.dense(x)

        if training:
            x = self.cosface([x, labels], training)
        else:
            x = self.cosface(x, training)

        return x
