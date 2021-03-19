from custom_layers import CosFace, ArcFace
from utils import EXTRA_LAYER_ACT_F

import tensorflow as tf
from tensorflow.keras import layers, activations
from transformers import TFBertModel


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


class ADBPretrainSoftmaxModel(tf.keras.Model):
    """Adaptive Decision Boundary with Softmax pre-training model."""

    def __init__(self, seq_len, num_classes):
        super(ADBPretrainSoftmaxModel, self).__init__()
        self.input_ids = layers.Input(shape=(seq_len))
        self.attention_mask = layers.Input(shape=(seq_len))
        self.token_type_ids = layers.Input(shape=(seq_len))

        self.bert = TFBertModel.from_pretrained('bert-base-uncased')
        hidden_size = self.bert.config.hidden_size  # 768
        hidden_dropout_prob = self.bert.config.hidden_dropout_prob  # 0.1

        self.dense = layers.Dense(hidden_size, activation=activations.relu)
        self.dropout = layers.Dropout(hidden_dropout_prob)
        self.dense2 = layers.Dense(num_classes, activation=activations.softmax)

    def call(self, inputs, training=None):
        input_ids, attention_mask, token_type_ids = inputs

        x = self.bert({'input_ids': input_ids, 'attention_mask': attention_mask, 'token_type_ids': token_type_ids})
        x = tf.reduce_mean(x.last_hidden_state, axis=1)
        x = self.dense(x)
        x = self.dropout(x)  # automatically applies only when training is True
        probs = self.dense2(x)

        if training:
            return probs

        return x  # return embeddings
