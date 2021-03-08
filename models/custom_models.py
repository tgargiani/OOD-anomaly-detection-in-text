from custom_layers import CosFace, ArcFace
from utils import EXTRA_LAYER_ACT_F

import tensorflow as tf
from tensorflow.keras import layers


class CosFaceModel(tf.keras.Model):
    def __init__(self, emb_dim, num_classes, extra_layer=False):
        super(CosFaceModel, self).__init__()
        self.extra_layer = extra_layer
        self.inp = layers.Input(shape=emb_dim)
        self.inp2 = layers.Input(shape=())

        if extra_layer:
            self.dense = layers.Dense(emb_dim, EXTRA_LAYER_ACT_F)

        self.cosface = CosFace(num_classes=num_classes)

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


class ArcFaceModel(tf.keras.Model):
    def __init__(self, emb_dim, num_classes, extra_layer=False):
        super(ArcFaceModel, self).__init__()
        self.extra_layer = extra_layer
        self.inp = layers.Input(shape=emb_dim)
        self.inp2 = layers.Input(shape=())

        if extra_layer:
            self.dense = layers.Dense(emb_dim, EXTRA_LAYER_ACT_F)

        self.arcface = ArcFace(num_classes=num_classes)

    def call(self, inputs, training=None):
        if training:
            x, labels = inputs
        else:
            x = inputs

        if self.extra_layer:
            x = self.dense(x)

        if training:
            x = self.arcface([x, labels], training)
        else:
            x = self.arcface(x, training)

        return x
