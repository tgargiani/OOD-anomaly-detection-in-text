from utils import compute_centroids, distance_metric

import numpy as np
import numpy.linalg as LA
import tensorflow as tf


def find_best_radius(X_train, y_train, centroids, step=0.01, constant=1.5):
    X, y = np.asarray(X_train), np.asarray(y_train)
    centroids = np.asarray(centroids)
    num_classes = len(set(y))  # number of classes
    radius = np.zeros(shape=num_classes)

    for c in range(num_classes):
        dists_sel = LA.norm(X - centroids[c], axis=1)  # distances of every point from the selected centroid

        while radius[c] < 2:  # maximum radius on a unit n-sphere is 2
            ood_mask = np.where(dists_sel > radius[c], 1, 0)  # out-of-domain
            id_mask = np.where(dists_sel <= radius[c], 1, 0)  # in-domain

            ood_criterion = (dists_sel - radius[c]) * ood_mask
            id_criterion = (radius[c] - dists_sel) * id_mask
            criterion = tf.reduce_mean(ood_criterion) - (tf.reduce_mean(id_criterion) * num_classes / constant)

            if criterion < 0:  # ID outweighs OOD
                radius[c] -= step
                break

            radius[c] += step

    return tf.convert_to_tensor(radius, dtype=tf.float32)


class ADBThreshold:
    """
    Adaptive Decision Boundary Threshold
    """

    def __init__(self):
        self.radius = None
        self.centroids = None
        self.oos_label = None

    def fit(self, X_train, y_train):
        X_train = tf.math.l2_normalize(X_train, axis=1)  # normalize to make sure it lies on a unit n-sphere
        self.centroids = compute_centroids(X_train, y_train)
        self.centroids = tf.math.l2_normalize(self.centroids, axis=1)
        self.radius = find_best_radius(X_train, y_train, self.centroids)

    def predict(self, X_test):
        X_test = tf.math.l2_normalize(X_test, axis=1)
        logits = distance_metric(X_test, self.centroids, 'euclidean')
        predictions = tf.math.argmin(logits, axis=1)

        c = tf.gather(self.centroids, predictions)
        d = tf.gather(self.radius, predictions)

        distance = tf.norm(X_test - c, ord='euclidean', axis=1)
        predictions = np.where(distance < d, predictions, self.oos_label)

        return predictions

    def predict_proba(self, X_test):
        raise NotImplementedError("Adaptive Decision Boundary Threshold can be used only in ood_train.")