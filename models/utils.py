import os, random, json
import tensorflow as tf
import numpy as np
import math
import matplotlib.pyplot as plt
import sklearn

RESULTS_PATH = os.path.join(os.path.dirname(__file__), '..', 'results')

RESULTS_CLINC150_PATH = os.path.join(RESULTS_PATH, 'CLINC150')
RESULTS_CLINC150_RIS_PATH = os.path.join(RESULTS_CLINC150_PATH, 'RIS')  # random intent selection results path

EMB_PATH = os.path.join(os.path.dirname(__file__), '..', 'embeddings')
DS_PATH = os.path.join(os.path.dirname(__file__), '..', 'datasets')

USE_DAN_PATH = os.path.join(EMB_PATH, 'universal-sentence-encoder_4')  # USE with DAN architecture
USE_TRAN_PATH = os.path.join(EMB_PATH, 'universal-sentence-encoder-large_5')  # USE with Transformer architecture

DS_CLINC150_PATH = os.path.join(DS_PATH, 'CLINC150')

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

    def get_X_y(self, lst, limit_num_sents, set_type: str):
        """
        Splits a part (contained in lst) of dataset into sentences and intents.

        :param:             lst - contains the dataset, list
                            limit_num_sents - specifies (if not None) the limited number of sentences per intent, int
                            set_type - deprecated; specifies the type of the received dataset (train, val or test), str
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

            if limit_num_sents and label != 'oos':  # don't limit number of OOD sentences
                if label not in label_occur_count.keys():
                    label_occur_count[label] = 0

                if label_occur_count[label] == limit_num_sents:  # skip sentence and label if reached limit
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


def visualize_2d_data(X, y, title, centroids=None, delta=None, save=False):
    fig, ax = plt.subplots(figsize=(6, 6))

    plt.title(title)
    ax.scatter(X[:, 0], X[:, 1], c=y)

    if (centroids is not None) and (delta is not None):
        for c, d in zip(centroids, delta):
            circle = plt.Circle(c, d, color='r', fill=False)
            ax.set_aspect('equal', adjustable='datalim')
            ax.add_patch(circle)

    if save:
        plt.savefig(f'{title}.pdf')
    else:
        plt.show()


def prepare_for_custom_triplet_loss_batches(X_train, y_train, batch_size, num_classes):
    """Reorders data so that each batch contains at least one random embedding per class."""

    num_samples = X_train.shape[0]
    num_samples_to_select = math.ceil(batch_size / num_classes)
    idx = 0
    num_seen_samples = 0

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

    return X_train, y_train


def get_intents_selection(lst, num_intents: int):
    """
    Returns a random selection of intent labels.
    :params:            lst - contains sublists in the following form: [sentence, label]
                        num_intents, int
    :returns:           selection, (num_intents, ) np.ndarray
    """

    unique_intents = list(set([l[1] for l in lst]))
    selection = np.random.choice(unique_intents, num_intents,
                                 replace=False)  # replace=False doesn't allow elements to repeat

    return selection


def get_filtered_lst(lst, selection):
    """
    Filters a list in order to contain only sublists with intent labels contained in selection.
    :returns:           filtered_lst, list
    """
    filtered_lst = [l for l in lst if l[1] in selection]

    return filtered_lst


def save_results(dataset_name, results_dct):
    if dataset_name == 'clinc150-data_full':
        path = os.path.join(RESULTS_CLINC150_RIS_PATH, 'results.json')
    else:
        raise ValueError("Wrong dataset name! Can't save.")

    with open(path, 'w') as f:
        json.dump(results_dct, f, indent=2)
