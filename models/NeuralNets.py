from AbstractNeuralNet import AbstractNeuralNet

import tensorflow as tf
from tensorflow.keras import layers, activations


class NeuralNet(AbstractNeuralNet):
    def create_model(self, emb_dim, num_classes):
        model = tf.keras.Sequential([
            layers.InputLayer(input_shape=emb_dim),
            layers.Dense(num_classes, activation=self.activation)])

        return model


class NeuralNetExtraLayer(AbstractNeuralNet):
    def create_model(self, emb_dim, num_classes):
        model = tf.keras.Sequential([
            layers.InputLayer(input_shape=emb_dim),
            layers.Dense(emb_dim, activation=activations.relu),
            layers.Dense(num_classes, activation=self.activation)])

        return model
