from AbstractNeuralNet import AbstractNeuralNet
from custom_models import CosFaceModel, ArcFaceModel
from utils import EXTRA_LAYER_ACT_F

import tensorflow as tf
from tensorflow.keras import layers, activations


class BaselineNN(AbstractNeuralNet):
    """Baseline Neural Net"""

    def create_model(self, emb_dim, num_classes):
        model = tf.keras.Sequential([
            layers.InputLayer(input_shape=emb_dim),
            layers.Dense(num_classes, activation=activations.softmax)])

        return model


class BaselineNNExtraLayer(AbstractNeuralNet):
    """Baseline Neural Net Extra Layer"""

    def create_model(self, emb_dim, num_classes):
        model = tf.keras.Sequential([
            layers.InputLayer(input_shape=emb_dim),
            layers.Dense(emb_dim, activation=EXTRA_LAYER_ACT_F),
            layers.Dense(num_classes, activation=activations.softmax)])

        return model


class CosFaceNN(AbstractNeuralNet):
    """
    CosFace Neural Net
    Based on https://arxiv.org/abs/1801.09414.
    """

    def create_model(self, emb_dim, num_classes):
        model = CosFaceModel(emb_dim, num_classes)

        return model


class CosFaceNNExtraLayer(AbstractNeuralNet):
    """
    CosFace Neural Net Extra Layer
    Based on https://arxiv.org/abs/1801.09414.
    """

    def create_model(self, emb_dim, num_classes):
        model = CosFaceModel(emb_dim, num_classes, extra_layer=True)

        return model


class CosFaceLOFNN(AbstractNeuralNet):
    """
    CosFace with Local Outlier Factor Neural Net
    Modified version of https://www.aclweb.org/anthology/P19-1548/.
    Used only in ood_train.
    """

    def create_model(self, emb_dim, num_classes):
        model = CosFaceModel(emb_dim, num_classes, extra_layer=True)

        return model


class ArcFaceNN(AbstractNeuralNet):
    """
    ArcFace Neural Net
    Based on https://arxiv.org/pdf/1801.07698.pdf.
    """

    def create_model(self, emb_dim, num_classes):
        model = ArcFaceModel(emb_dim, num_classes)

        return model


class ArcFaceNNExtraLayer(AbstractNeuralNet):
    """
    ArcFace Neural Net Extra Layer
    Based on https://arxiv.org/pdf/1801.07698.pdf.
    """

    def create_model(self, emb_dim, num_classes):
        model = ArcFaceModel(emb_dim, num_classes, extra_layer=True)

        return model
