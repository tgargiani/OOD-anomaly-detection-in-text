import os, random
import tensorflow as tf
import numpy as np
import math

EMB_PATH = os.path.join(os.path.dirname(__file__), '..', 'embeddings')
DS_PATH = os.path.join(os.path.dirname(__file__), '..', 'datasets')

USE_DAN_PATH = os.path.join(EMB_PATH, 'universal-sentence-encoder_4')  # USE with DAN architecture
USE_TRAN_PATH = os.path.join(EMB_PATH, 'universal-sentence-encoder-large_5')  # USE with Transformer architecture

DS_CLINC150_PATH = os.path.join(DS_PATH, 'CLINC150')

NUM_SENTS = {'train': 18, 'val': 18, 'test': 30, 'train_oos': 20, 'val_oos': 20, 'test_oos': 60}

EXTRA_LAYER_ACT_F = tf.keras.activations.relu  # specifies the activation function of the extra layer in NNs
NEEDS_VAL = ['BaselineNN', 'BaselineNNExtraLayer', 'CosFaceNN', 'CosFaceNNExtraLayer', 'CosFaceLOFNN', 'ArcFaceNN',
             'ArcFaceNNExtraLayer']  # names of models that need validation splits, it's the majority of NNs


class Split:
    """
    Class used when splitting the training, validation and test set.

    :attributes:            intents_dct - keys: intent labels, values: unique ids, dict
                            new_key_value - keeps track of the newest unique id for intents_dct, int
                            embed_f - function that encodes sentences as embeddings
    """

    def __init__(self, embed_f):
        self.intents_dct = {}
        self.new_key_value = 0
        self.embed_f = embed_f

    def get_X_y(self, lst, limit_num_sents: bool, set_type: str):
        """
        Splits a part (contained in lst) of dataset into sentences and intents.

        :param:             lst - contains the dataset, list
                            limit_num_sents - specifies if every intent should have a limited number of sentences, bool
                            set_type - specifies the type of the received dataset (train, val or test), str
        :returns:           X - sentences encoded as embeddings, tf.Tensor OR sentences, list
                            y - intents, tf.Tensor
        """

        X = []
        y = []

        if limit_num_sents:  # these aren't needed normally
            random.shuffle(lst)
            label_occur_count = {}

        for sent, label in lst:
            if label not in self.intents_dct.keys():
                self.intents_dct[label] = self.new_key_value
                self.new_key_value += 1

            if limit_num_sents:
                if label not in label_occur_count.keys():
                    label_occur_count[label] = 0

                # limit of occurrence of specific intent:
                occur_limit = NUM_SENTS[set_type] if label != 'oos' else NUM_SENTS[f'{set_type}_oos']

                if label_occur_count[label] == occur_limit:  # skip sentence and label if reached limit
                    continue

                label_occur_count[label] += 1

            X.append(sent)
            y.append(self.intents_dct[label])

        if self.embed_f is not None:
            X = self.embed_f(X)
            X = tf.convert_to_tensor(X, dtype='float32')

        y = tf.convert_to_tensor(y, dtype='int32')

        return X, y


def print_results(dataset_name: str, model_name: str, emb_name: str, results_dct: dict):
    """Helper print function."""

    print(f'dataset_name: {dataset_name}, model_name: {model_name}, embedding_name: {emb_name} -- {results_dct}\n')


def find_best_threshold(val_predictions_labels, oos_label):
    """
    Function used to find the best threshold in oos-threshold.
    :param:            val_predictions_labels - prediction on the validation set, list
                        oos_label - encodes oos label, int
    :returns:           threshold - best threshold
    """

    # Initialize search for best threshold
    thresholds = np.linspace(0, 1, 101)
    previous_val_accuracy = 0
    threshold = 0

    # Find best threshold
    for idx, tr in enumerate(thresholds):
        val_accuracy_correct = 0
        val_accuracy_out_of = 0

        for pred, true_label in val_predictions_labels:
            pred_label = pred[0]
            similarity = pred[1]

            if similarity < tr:
                pred_label = oos_label

            if pred_label == true_label:
                val_accuracy_correct += 1

            val_accuracy_out_of += 1

        val_accuracy = val_accuracy_correct / val_accuracy_out_of

        if val_accuracy < previous_val_accuracy:
            threshold = thresholds[idx - 1]  # best threshold is the previous one
            break

        previous_val_accuracy = val_accuracy
        threshold = tr

    return threshold


def compute_centroids(X, y):
    X = np.asarray(X)
    y = np.asarray(y)

    emb_dim = X.shape[1]
    classes = set(y)
    num_classes = len(classes)

    centroids = np.zeros(shape=(num_classes, emb_dim))

    for label in range(num_classes):
        embeddings = X[y == label]
        num_embeddings = len(embeddings)

        for emb in embeddings:
            centroids[label, :] += emb

        centroids[label, :] /= num_embeddings

    return tf.convert_to_tensor(centroids, dtype=tf.float32)


def distance_metric(X, centroids, dist_type):
    X = np.asarray(X)
    centroids = np.asarray(centroids)

    num_embeddings = X.shape[0]
    num_centroids = centroids.shape[0]  # equivalent to num_classes

    if dist_type == 'euclidean':
        # modify arrays to shape (num_embeddings, num_centroids, emb_dim) in order to compare them
        x = np.repeat(X[:, np.newaxis, :], repeats=num_centroids, axis=1)
        centroids = np.repeat(centroids[np.newaxis, :, :], repeats=num_embeddings, axis=0)

        logits = tf.norm(x - centroids, ord='euclidean', axis=2)
    else:
        x_norm = tf.nn.l2_normalize(X, axis=1)
        centroids_norm = tf.nn.l2_normalize(centroids, axis=1)
        cos_sim = tf.matmul(x_norm, tf.transpose(centroids_norm))

        if dist_type == 'cosine':
            logits = 1 - cos_sim
        else:  # angular
            logits = tf.math.acos(cos_sim) / math.pi

    return tf.convert_to_tensor(logits)


def batches(lst, batch_size):
    for i in range(0, len(lst), batch_size):
        yield lst[i:i + batch_size]
