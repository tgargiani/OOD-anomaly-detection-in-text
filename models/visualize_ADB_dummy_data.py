from custom_models import ADBPretrainSoftmaxModel, ADBPretrainCosFaceModel, ADBPretrainTripletLossModel
from utils import batches

from NeuralNets import AdaptiveDecisionBoundaryNN
from sklearn.datasets import make_blobs, make_classification
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow import optimizers, losses
import numpy as np


def visualize_data(X, y, title, centroids=None, delta=None):
    fig, ax = plt.subplots(figsize=(6, 6))
    plt.title(title)
    ax.scatter(X[:, 0], X[:, 1], c=y)

    if (centroids is not None) and (delta is not None):
        for c, d in zip(centroids, delta):
            circle = plt.Circle(c, d, color='r', fill=False)
            ax.set_aspect('equal', adjustable='datalim')
            ax.add_patch(circle)

    plt.show()


def embed_f(X, pretraining_model):
    embeddings_lst = []

    for batch in batches(X, 32):  # iterate in batches of size 32
        temp_emb = pretraining_model(batch)
        embeddings_lst.append(temp_emb)

    embeddings = tf.concat(embeddings_lst, axis=0)

    return embeddings


if __name__ == '__main__':
    emb_dim = 2
    num_classes = 2
    type = 'triplet_loss'  # softmax, cosface or triplet_loss

    # Create 2D dummy train data
    # Moi
    # X_train, y_train = make_blobs(n_samples=500, n_features=emb_dim, centers=num_classes, cluster_std=2.5)

    # Petr
    X1, Y1 = make_classification(n_classes=num_classes, n_features=emb_dim, n_redundant=0, n_informative=emb_dim,
                                 n_clusters_per_class=2)
    X2, Y2 = make_classification(n_classes=num_classes, n_features=emb_dim, n_redundant=0, n_informative=emb_dim,
                                 n_clusters_per_class=2)
    X_train = np.concatenate((X1, X2), axis=0)
    y_train = np.concatenate((Y1, Y2), axis=0)

    # Visualize data
    visualize_data(X_train, y_train, f'Points - {type}')

    # Train pre-training model
    if type == 'softmax':
        pretraining_model = ADBPretrainSoftmaxModel(emb_dim, num_classes)
    elif type == 'cosface':
        pretraining_model = ADBPretrainCosFaceModel(emb_dim, num_classes)
    else:  # triplet_loss
        pretraining_model = ADBPretrainTripletLossModel(emb_dim)

    if type in ['softmax', 'cosface']:
        loss = losses.SparseCategoricalCrossentropy()
        shuffle = True  # default
        batch_size = None  # defaults to 32
    else:  # triplet_loss
        loss = tfa.losses.TripletSemiHardLoss()
        shuffle = True  # shuffle before every epoch in order to guarantee diversity in pos and neg samples
        batch_size = 256  # same as above

    pretraining_model.compile(optimizer=optimizers.Adam(learning_rate=2e-5), loss=loss, metrics=['accuracy'])

    if type in ['softmax', 'triplet_loss']:
        X = X_train
    else:  # cosface
        X = [X_train, y_train]

    pretraining_model.fit(X, y_train, epochs=40, shuffle=shuffle, batch_size=batch_size)

    X_train_pretr = embed_f(X_train, pretraining_model)

    adb_distance = 'euclidean'
    model = AdaptiveDecisionBoundaryNN(adb_distance)
    model.fit(X_train_pretr, y_train)

    centroids = model.centroids
    delta = tf.convert_to_tensor(model.delta)
    visualize_data(X_train_pretr, y_train, f'Points after pre-training and radius - {type} - {adb_distance} distance',
                   centroids=centroids,
                   delta=delta)
