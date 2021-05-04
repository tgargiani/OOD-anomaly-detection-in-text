from utils import EXTRA_LAYER_ACT_F

from abc import ABC, abstractmethod
import tensorflow as tf
from tensorflow.keras import losses, layers
import numpy as np
from sklearn.neighbors import LocalOutlierFactor


class AbstractNeuralNet(ABC):
    def __init__(self, loss=losses.SparseCategoricalCrossentropy()):
        tf.random.set_seed(7)  # set seed in order to have reproducible results

        self.model = None
        self.model_name = type(self).__name__
        self.loss = loss
        self.oos_label = None  # used for CosFaceLofNN

    @abstractmethod
    def create_model(self, emb_dim, num_classes):
        raise NotImplementedError('You have to create a model.')

    def fit(self, X_train, y_train, X_val, y_val):
        emb_dim = X_train.shape[1]  # embedding dimension
        num_classes = len(set(np.asarray(y_train)))  # number of classes

        self.model = self.create_model(emb_dim, num_classes)

        self.model.compile(optimizer='adam',
                           loss=self.loss,
                           metrics=['accuracy'])

        if self.model_name in ['CosFaceNN', 'CosFaceNNExtraLayer', 'CosFaceLOFNN', 'ArcFaceNN', 'ArcFaceNNExtraLayer']:
            X_train = [X_train, y_train]

        self.model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=40)

        if self.model_name == 'CosFaceLOFNN':
            X_train = X_train[0]
            weights = self.model.layers[0].get_weights()  # weights of Dense layer

            self.feat_model = tf.keras.Sequential([  # same as self.model except last layer (CosFace)
                layers.Input(emb_dim),
                layers.Dense(emb_dim, EXTRA_LAYER_ACT_F)])

            self.feat_model.layers[0].set_weights(weights)

            self.lof = LocalOutlierFactor(novelty=True, n_jobs=-1)

            X_lof_train = self.feat_model.predict(X_train)  # extract discriminative features
            self.lof.fit(X_lof_train)

    def predict(self, X_test):
        """Returns predictions with class labels."""

        probs = self.model.predict(X_test)
        predictions = np.argmax(probs, axis=1)

        if self.model_name == 'CosFaceLOFNN':
            X_lof_test = self.feat_model.predict(X_test)
            lof_preds = self.lof.predict(X_lof_test)

            predictions = np.where(lof_preds == 1, predictions, self.oos_label)

        return predictions

    def predict_proba(self, X_test):
        """Returns probabilities of each label."""

        if self.model_name == 'CosFaceLOFNN':
            raise NotImplementedError("CosFace with Local Outlier Factor Neural Net can be used only in ood_train.")

        probs = self.model.predict(X_test)

        return probs
