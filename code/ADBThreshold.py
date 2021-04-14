from utils import compute_centroids, distance_metric

import numpy as np
import numpy.linalg as LA
import tensorflow as tf


def find_best_radius(X_train, y_train, centroids, step=0.01):
    X, y = np.asarray(X_train), np.asarray(y_train)
    centroids = np.asarray(centroids)
    num_classes = len(set(y))  # number of classes
    radius = np.zeros(shape=num_classes)

    for c in range(num_classes):
        dists_sel = LA.norm(X - centroids[c], axis=1)  # distances of every point from the selected centroid
        # accuracy_previous = 0

        while True:
            pos_mask = np.where(dists_sel > radius[c], 1, 0)  # OOD
            neg_mask = np.where(dists_sel <= radius[c], 1, 0)  # ID

            pos_loss = (dists_sel - radius[c]) * pos_mask
            neg_loss = (radius[c] - dists_sel) * neg_mask
            loss = tf.reduce_mean(pos_loss) - tf.reduce_mean(neg_loss) * num_classes / 1.5
            # loss = tf.reduce_mean(dists_sel - radius[c]) # equivalent to: tf.reduce_mean(pos_loss) - tf.reduce_mean(neg_loss)

            if loss < 0:  # ID outweighs OOD
                radius[c] -= step
                break

            # id_true_labels = y[dists_sel < radius[c]]  # labels of the embeddings considered as ID
            #
            # accuracy_correct = 0
            # accuracy_out_of = 0
            #
            # for label in id_true_labels:
            #     if label == c:
            #         accuracy_correct += 1
            #
            #     accuracy_out_of += 1
            #
            # accuracy = round(accuracy_correct / accuracy_out_of, 1) if accuracy_out_of != 0 else 0
            #
            # if accuracy < accuracy_previous:
            #     radius[c] -= step  # best radius is the previous one
            #     break
            #
            # accuracy_previous = accuracy

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
        self.centroids = compute_centroids(X_train, y_train)
        self.radius = find_best_radius(X_train, y_train, self.centroids)

    def predict(self, X_test):
        logits = distance_metric(X_test, self.centroids, 'euclidean')
        predictions = tf.math.argmin(logits, axis=1)

        c = tf.gather(self.centroids, predictions)
        d = tf.gather(self.radius, predictions)

        distance = tf.norm(X_test - c, ord='euclidean', axis=1)
        predictions = np.where(distance < d, predictions, self.oos_label)

        return predictions

    def predict_proba(self, X_test):
        raise NotImplementedError("Adaptive Decision Boundary Threshold can be used only in ood_train.")


if __name__ == '__main__':
    # x = np.array([[0.5, 0.5], [1, 0.5], [0.6, 0.4], [-0.5, -0.5], [-1, -0.5], [-0.74, -0.37]])
    # x = np.array([[0.5, 0.5], [1, 0.5], [-0.5, -0.5], [-1, -0.5]])
    x = np.array([[0.5, 0.5], [1, 0.5], [-0.5, -0.5], [-1, -0.5], [-0.74, -0.37]])
    y = np.array([0, 0, 1, 1, 1])

    adb = ADBThreshold()
    adb.fit(x, y)
    print(adb.predict(np.array([[0.6, 0.6], [-0.6, -0.6], [10, 10]])))
