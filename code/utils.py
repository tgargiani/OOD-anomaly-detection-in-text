import os, random, json
import tensorflow as tf
import numpy as np
import math
import matplotlib.pyplot as plt
import sklearn
from sklearn.model_selection import StratifiedKFold, train_test_split

RESULTS_PATH = os.path.join(os.path.dirname(__file__), '..', 'results')

RESULTS_CLINC150_PATH = os.path.join(RESULTS_PATH, 'CLINC150')
RESULTS_CLINC150_RIS_PATH = os.path.join(RESULTS_CLINC150_PATH, 'RIS')  # random intent selection results path

EMB_PATH = os.path.join(os.path.dirname(__file__), '..', 'embeddings')
DS_PATH = os.path.join(os.path.dirname(__file__), '..', 'datasets')

USE_DAN_PATH = os.path.join(EMB_PATH, 'universal-sentence-encoder_4')  # USE with DAN architecture
USE_TRAN_PATH = os.path.join(EMB_PATH, 'universal-sentence-encoder-large_5')  # USE with Transformer architecture

DS_CLINC150_PATH = os.path.join(DS_PATH, 'CLINC150')
DS_ROSTD_PATH = os.path.join(DS_PATH, 'ROSTD')
DS_LL_PATH = os.path.join(DS_PATH, 'lucid_lindia-2')

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
    """
    Reorders data so that each batch contains at least one random embedding per class.
    Not used - Triplet Loss requires at least two embeddings per class.
    """

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


def save_results(results_dct, dataset_name, model_name):
    if dataset_name == 'clinc150-data_full':
        path = os.path.join(RESULTS_CLINC150_RIS_PATH, f'{model_name}_results.json')
    else:
        raise ValueError("Wrong dataset name! Can't save.")

    with open(path, 'w') as f:
        json.dump(results_dct, f, indent=2)


def get_unsplit_Xy_ID_OOD(dialogue_path):
    category = dialogue_path.split(sep=os.sep)[-2]
    ood_cross_path = os.path.join(DS_LL_PATH, 'data_cross_ood', f'{category}_ood.json')

    with open(dialogue_path) as f:
        dialogue = json.load(f)

    # out-of-domain
    Xy_OOD = []

    with open(ood_cross_path) as f:
        ood_cross = json.load(f)

    for sent in ood_cross['ood']:
        Xy_OOD.append([sent, 'oos'])

    # in-domain global
    Xy_ID_global = []

    for intent in dialogue['globalIntents']:
        for sent in dialogue['globalIntents'][intent]['original_plus_noised']:
            Xy_ID_global.append([sent, intent])

    # in-domain local
    intents_decision_nodes = {}

    for node in dialogue['decisionNodes']:
        Xy_ID = []

        for intent in dialogue['links'][str(node)]:
            intent_str = str(intent)

            if intent_str not in dialogue['intents'].keys():
                continue

            for sent in dialogue['intents'][intent_str]['original_plus_noised']:
                Xy_ID.append([sent, intent])

        Xy_ID += Xy_ID_global

        intents_decision_nodes[node] = (Xy_ID, Xy_OOD)

    return intents_decision_nodes


def cross_val_evaluate(categories, evaluate, model, model_name, emb_name, embed_f, limit_num_sents):
    original_emb_name = emb_name
    dct_results_lst = []
    total_time_pretraining = 0

    if original_emb_name == 'placeholder':
        import tensorflow_hub as hub
        from custom_embeddings import create_embed_f

        use_embed = hub.load(USE_DAN_PATH)
        emb_name = 'use_dan_cosface'

        # use_embed = hub.load(USE_TRAN_PATH)
        # emb_name = 'use_tran_cosface'

    for cat in categories:
        cat_path = os.path.join(DS_LL_PATH, 'data', cat)
        dataset_paths = [os.path.join(cat_path, ds) for ds in os.listdir(cat_path)]

        for dialogue_path in dataset_paths:
            intents_decision_nodes = get_unsplit_Xy_ID_OOD(dialogue_path)

            for (Xy_ID, Xy_OOD) in intents_decision_nodes.values():
                y_ID = [x[1] for x in Xy_ID]
                y_OOD = [x[1] for x in Xy_OOD]

                dataset = {}
                skf = StratifiedKFold(n_splits=5)

                for (train_idx_ID, test_idx_ID), (train_idx_OOD, test_idx_OOD) in zip(skf.split(Xy_ID, y_ID),
                                                                                      skf.split(Xy_OOD, y_OOD)):
                    train_idx_ID, val_idx_ID = train_test_split(train_idx_ID, test_size=0.2)
                    dataset['train'] = [Xy_ID[i] for i in train_idx_ID]
                    dataset['val'] = [Xy_ID[i] for i in val_idx_ID]
                    dataset['test'] = [Xy_ID[i] for i in test_idx_ID]

                    train_idx_OOD, val_idx_OOD = train_test_split(train_idx_OOD, test_size=0.2)
                    dataset['oos_train'] = [Xy_OOD[i] for i in train_idx_OOD]
                    dataset['oos_val'] = [Xy_OOD[i] for i in val_idx_OOD]
                    dataset['oos_test'] = [Xy_OOD[i] for i in test_idx_OOD]

                    if original_emb_name == 'placeholder':
                        embed_f, time_pretraining = create_embed_f(use_embed, dataset, limit_num_sents, type='cosface')
                        total_time_pretraining += time_pretraining

                    results_dct = evaluate(dataset, model, model_name, embed_f, limit_num_sents)
                    dct_results_lst.append(results_dct)

    results_dct = {}
    num_results = len(dct_results_lst)

    for dct in dct_results_lst:
        for key in dct:
            if key not in results_dct:
                results_dct[key] = 0

            results_dct[key] += dct[key]

    for key in results_dct:
        if key not in ['time_train', 'time_inference']:  # keep track of total train and inference time
            results_dct[key] /= num_results

        results_dct[key] = round(results_dct[key], 1)

    if total_time_pretraining != 0:
        results_dct['time_pretraining'] = round(total_time_pretraining, 1)

    return results_dct, emb_name
