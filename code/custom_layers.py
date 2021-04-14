import tensorflow as tf
from tensorflow.keras import layers, activations
import tensorflow.keras.backend as K
import math


class CosFace(layers.Layer):
    """
    Implementation of CosFace layer. Reference: https://arxiv.org/abs/1801.09414
    Credits: https://www.kaggle.com/chankhavu/keras-layers-arcface-cosface-adacos

    Arguments:
      num_classes: number of classes to classify
      s: scale factor
      m: margin
      regularizer: weights regularizer
    """

    def __init__(self,
                 num_classes,
                 s=64.0,
                 m=0.1,
                 regularizer=None,
                 name='cosface',
                 **kwargs):

        super().__init__(name=name, **kwargs)
        self._n_classes = num_classes
        self._s = float(s)
        self._m = float(m)
        self._regularizer = regularizer

    def build(self, input_shape):
        embedding_shape, label_shape = input_shape
        self._w = self.add_weight(shape=(embedding_shape[-1], self._n_classes),
                                  initializer='glorot_uniform',
                                  trainable=True,
                                  regularizer=self._regularizer)

    def call(self, inputs, training=None):
        """
        During training, requires 2 inputs: embedding (after backbone+pool+dense),
        and ground truth labels. The labels should be sparse (and use
        sparse_categorical_crossentropy as loss).
        """

        if training:
            embeddings, labels = inputs
            # Squeezing is necessary for Keras. It expands the dimension to (n, 1)
            labels = tf.reshape(labels, [-1], name='label_shape_correction')
        else:
            embeddings = inputs

        # Normalize features and weights and compute dot product
        x = tf.nn.l2_normalize(embeddings, axis=1, name='normalize_prelogits')
        w = tf.nn.l2_normalize(self._w, axis=0, name='normalize_weights')
        cosine_sim = tf.matmul(x, w, name='cosine_similarity')

        if training:
            one_hot_labels = tf.one_hot(labels,
                                        depth=self._n_classes,
                                        name='one_hot_labels')
            final_theta = tf.where(tf.cast(one_hot_labels, dtype=tf.bool),
                                   cosine_sim - self._m,
                                   cosine_sim,
                                   name='cosine_sim_with_margin')
            output = self._s * final_theta
        else:
            # We don't have labels if we're not in training mode
            output = self._s * cosine_sim

        return tf.nn.softmax(output)


class ArcFace(layers.Layer):
    """
    Implementation of ArcFace layer. Reference: https://arxiv.org/abs/1801.07698
    Credits: https://www.kaggle.com/chankhavu/keras-layers-arcface-cosface-adacos

    Arguments:
      num_classes: number of classes to classify
      s: scale factor
      m: margin
      regularizer: weights regularizer
    """

    def __init__(self,
                 num_classes,
                 s=30.0,
                 m=0.5,
                 regularizer=None,
                 name='arcface',
                 **kwargs):

        super().__init__(name=name, **kwargs)
        self._n_classes = num_classes
        self._s = float(s)
        self._m = float(m)
        self._regularizer = regularizer

    def build(self, input_shape):
        embedding_shape, label_shape = input_shape
        self._w = self.add_weight(shape=(embedding_shape[-1], self._n_classes),
                                  initializer='glorot_uniform',
                                  trainable=True,
                                  regularizer=self._regularizer,
                                  name='cosine_weights')

    def call(self, inputs, training=None):
        """
        During training, requires 2 inputs: embedding (after backbone+pool+dense),
        and ground truth labels. The labels should be sparse (and use
        sparse_categorical_crossentropy as loss).
        """

        if training:
            embeddings, labels = inputs
            # Squeezing is necessary for Keras. It expands the dimension to (n, 1)
            labels = tf.reshape(labels, [-1], name='label_shape_correction')
        else:
            embeddings = inputs

        # Normalize features and weights and compute dot product
        x = tf.nn.l2_normalize(embeddings, axis=1, name='normalize_prelogits')
        w = tf.nn.l2_normalize(self._w, axis=0, name='normalize_weights')
        cosine_sim = tf.matmul(x, w, name='cosine_similarity')

        if training:
            one_hot_labels = tf.one_hot(labels,
                                        depth=self._n_classes,
                                        name='one_hot_labels')
            theta = tf.math.acos(K.clip(
                cosine_sim, -1.0 + K.epsilon(), 1.0 - K.epsilon()))
            selected_labels = tf.where(tf.greater(theta, math.pi - self._m),
                                       tf.zeros_like(one_hot_labels),
                                       one_hot_labels,
                                       name='selected_labels')
            final_theta = tf.where(tf.cast(selected_labels, dtype=tf.bool),
                                   theta + self._m,
                                   theta,
                                   name='final_theta')
            output = tf.math.cos(final_theta, name='cosine_sim_with_margin')
        else:
            # We don't have labels if we're not in training mode
            output = self._s * cosine_sim

        return tf.nn.softmax(output)


class AdaptiveDecisionBoundary(layers.Layer):
    """
    Implementation of ADB as last layer that directly computes loss. Compile model with loss=None.
    Reference: https://arxiv.org/pdf/2012.10209.pdf.
    Code ported from: https://github.com/thuiar/Adaptive-Decision-Boundary.
    Inspiration: https://stackoverflow.com/questions/64223840/use-additional-trainable-variables-in-keras-tensorflow-custom-loss-function.
    """

    def __init__(self, num_classes, centroids, dist_type):
        super().__init__()
        self.delta = tf.Variable(tf.random.normal([num_classes]), trainable=True)
        self.centroids = tf.Variable(centroids, trainable=False, dtype=tf.float32)
        self.dist_type = dist_type

    def custom_loss(self, embeddings, labels):
        labels = tf.cast(labels, dtype=tf.int32)  # TF has automatically casted to tf.float32, revert back
        soft_delta = activations.softplus(self.delta)

        # c = centroids[labels] # can't do this because eager execution isn't enabled
        # d = soft_delta[labels]
        c = tf.gather(self.centroids, labels)
        d = tf.gather(soft_delta, labels)

        if self.dist_type == 'euclidean':
            distance = tf.norm(embeddings - c, ord='euclidean', axis=1)
        else:
            embeddings_norm = tf.nn.l2_normalize(embeddings, axis=1)
            c = tf.squeeze(c)
            c_norm = tf.nn.l2_normalize(c, axis=1)
            cos_sim = tf.matmul(embeddings_norm, tf.transpose(c_norm))

            if self.dist_type == 'cosine':
                distance = 1 - cos_sim
            else:  # angular
                distance = tf.math.acos(cos_sim) / math.pi

            distance = tf.linalg.diag_part(distance)

        pos_mask = tf.cast([distance >= d], dtype=tf.float32)
        neg_mask = tf.cast([distance < d], dtype=tf.float32)

        pos_loss = (distance - d) * pos_mask * 500  # multiplication introduced by TG
        neg_loss = (d - distance) * neg_mask
        loss = tf.reduce_mean(pos_loss) + tf.reduce_mean(neg_loss)

        return loss

    def call(self, inputs):
        # Layer (and model) used only for training, so it will always receive the same inputs.
        # The whole loss is computed in this layer, therefore model should be compiled with loss=None.

        embeddings, labels = inputs
        loss = self.custom_loss(embeddings, labels)
        self.add_loss(loss)

        return loss
