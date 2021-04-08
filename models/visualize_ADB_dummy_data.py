from custom_models import ADBPretrainSoftmaxModel, ADBPretrainCosFaceModel, ADBPretrainTripletLossModel
from utils import batches, visualize_2d_data

from NeuralNets import AdaptiveDecisionBoundaryNN
from sklearn.datasets import make_blobs, make_classification
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow import optimizers, losses
import numpy as np
import sklearn
import math


def embed_f(X, pretraining_model):
    embeddings_lst = []

    for batch in batches(X, 32):  # iterate in batches of size 32
        temp_emb = pretraining_model(batch)
        embeddings_lst.append(temp_emb)

    embeddings = tf.concat(embeddings_lst, axis=0)

    return embeddings


if __name__ == '__main__':
    # tf.random.set_seed(7)
    # np.random.seed(7)

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
    visualize_2d_data(X_train, y_train, f'Points - {type}')

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
        # shuffle = True  # shuffle before every epoch in order to guarantee diversity in pos and neg samples
        # batch_size = 256  # same as above

        shuffle = False  # shuffle manually
        batch_size = 300
        num_samples = X_train.shape[0]
        num_samples_to_select = math.ceil(batch_size / num_classes)
        idx = 0
        num_seen_samples = 0

        # create custom batches - each batch should contain at least one random embedding per class
        X, y = np.asarray(X_train), np.asarray(y_train)
        X, y = sklearn.utils.shuffle(X, y)
        Xy = list(zip(X, y))
        Xy = sorted(Xy, key=lambda x: x[1])
        Xy = np.asarray(Xy)

        X_train = np.empty(shape=X_train.shape)
        y_train = np.empty(shape=y_train.shape)

        while num_seen_samples < num_samples:
            to_idx = idx + num_samples_to_select

            for c in range(num_classes):
                X_selection = np.stack(Xy[:, 0][Xy[:, 1] == c][idx:to_idx])
                y_selection = np.stack(Xy[:, 1][Xy[:, 1] == c][idx:to_idx])
                num_selected = len(y_selection)

                X_train[num_seen_samples:num_seen_samples + num_selected, :] = X_selection
                y_train[num_seen_samples:num_seen_samples + num_selected] = y_selection

                num_seen_samples += num_selected

            idx = to_idx

        X_train = tf.convert_to_tensor(X_train)
        y_train = tf.convert_to_tensor(y_train)

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
    visualize_2d_data(X_train_pretr, y_train,
                      f'Points after pre-training and radius - {type} - {adb_distance} distance',
                      centroids=centroids,
                      delta=delta)
