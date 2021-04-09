from AbstractNeuralNet import AbstractNeuralNet
from custom_models import CosFaceModel, ArcFaceModel
from custom_layers import AdaptiveDecisionBoundary
from utils import EXTRA_LAYER_ACT_F, compute_centroids, distance_metric

import tensorflow as tf
from tensorflow.keras import layers, activations
import numpy as np
import math


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


class AdaptiveDecisionBoundaryNN:
    """
    Adaptive Decision Boundary Neural Net
    Based on https://arxiv.org/pdf/2012.10209.pdf.
    """

    def __init__(self, dist_type: str):
        tf.random.set_seed(7)  # set seed in order to have reproducible results

        self.model = None
        self.model_name = type(self).__name__
        self.delta = None
        self.centroids = None
        self.oos_label = None
        self.dist_type = dist_type  # euclidean, cosine or angular

    def fit(self, X_train, y_train):
        num_embeddings, emb_dim = X_train.shape  # number of embeddings, embedding dimension
        num_classes = len(set(np.asarray(y_train)))  # number of classes
        self.centroids = compute_centroids(X_train, y_train)

        embedding_input = layers.Input(shape=(emb_dim))
        label_input = layers.Input(shape=(1))
        dense_output = layers.Dense(emb_dim, activation=activations.relu)(embedding_input)
        dense_output = layers.Dense(emb_dim, activation=activations.relu)(dense_output)
        dense_output = layers.Dense(emb_dim, activation=activations.relu)(dense_output)
        loss = AdaptiveDecisionBoundary(num_classes, self.centroids, self.dist_type)((dense_output, label_input))
        self.model = tf.keras.Model(inputs=[embedding_input, label_input], outputs=loss)

        self.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.005), loss=None)
        self.model.fit([X_train, y_train], None, epochs=10)

        self.delta = self.model.layers[-1].delta
        self.delta = activations.softplus(self.delta)

    def predict(self, X_test):
        logits = distance_metric(X_test, self.centroids, self.dist_type)
        predictions = tf.math.argmin(logits, axis=1)

        c = tf.gather(self.centroids, predictions)
        d = tf.gather(self.delta, predictions)

        if self.dist_type == 'euclidean':
            distance = tf.norm(X_test - c, ord='euclidean', axis=1)
        else:
            X_test_norm = tf.nn.l2_normalize(X_test, axis=1)
            c_norm = tf.nn.l2_normalize(c, axis=1)
            cos_sim = tf.matmul(X_test_norm, tf.transpose(c_norm))

            if self.dist_type == 'cosine':
                distance = 1 - cos_sim
            else:  # angular
                distance = tf.math.acos(cos_sim) / math.pi

            distance = tf.linalg.diag_part(distance)

        predictions = np.where(distance < d, predictions, self.oos_label)

        return predictions

    def predict_proba(self, X_test):
        raise NotImplementedError("Adaptive Decision Boundary Neural Net can be used only in ood_train.")
