from utils import Split, batches, visualize_2d_data
from custom_models import ADBPretrainBERTSoftmaxModel, ADBPretrainBERTCosFaceModel, ADBPretrainBERTTripletLossModel, \
    ADBPretrainSoftmaxModel, ADBPretrainCosFaceModel, ADBPretrainTripletLossModel

import numpy as np
from transformers import AutoTokenizer
import tensorflow as tf
from tensorflow.keras import losses, optimizers
import tensorflow_addons as tfa
import sklearn
from sklearn.decomposition import PCA
import math


def create_bert_embed_f(dataset_train, limit_num_sents, type: str):
    """Fine-tunes embeddings from BERT. Returns new embed function."""

    SEQ_LEN = 64  # sequence length of BERT
    split = Split(None)  # split without changing to embedding
    X_train, y_train = split.get_X_y(dataset_train, limit_num_sents=limit_num_sents, set_type='train')

    num_classes = len(set(np.asarray(y_train)))
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

    train_encoded_batch = tokenizer(X_train, padding='max_length', max_length=SEQ_LEN, truncation=True,
                                    return_attention_mask=True, return_tensors='tf')
    train_input_ids = train_encoded_batch.input_ids
    train_attention_mask = train_encoded_batch.attention_mask
    train_token_type_ids = train_encoded_batch.token_type_ids

    if type == 'softmax':
        model = ADBPretrainBERTSoftmaxModel(SEQ_LEN, num_classes)
    elif type == 'cosface':
        model = ADBPretrainBERTCosFaceModel(SEQ_LEN, num_classes)
    else:  # triplet_loss
        model = ADBPretrainBERTTripletLossModel(SEQ_LEN)

    if type in ['softmax', 'cosface']:
        loss = losses.SparseCategoricalCrossentropy()
        shuffle = True  # default
        batch_size = None  # defaults to 32
    else:  # triplet_loss
        loss = tfa.losses.TripletSemiHardLoss()
        shuffle = True  # shuffle before every epoch in order to guarantee diversity in pos and neg samples
        batch_size = 64  # same as above; larger batch size results in OOM error

    model.compile(optimizer=optimizers.Adam(learning_rate=2e-5), loss=loss, metrics=['accuracy'])

    if type in ['softmax', 'triplet_loss']:
        X = [train_input_ids, train_attention_mask, train_token_type_ids]
    else:  # cosface
        X = [train_input_ids, train_attention_mask, train_token_type_ids, y_train]

    model.fit(X, y_train, epochs=2, shuffle=shuffle, batch_size=batch_size)

    def embed_f(X):
        embeddings_lst = []

        for batch in batches(X, 32):  # iterate in batches of size 32
            encoded_batch = tokenizer(batch, padding='max_length', max_length=SEQ_LEN, truncation=True,
                                      return_attention_mask=True, return_tensors='tf')
            input_ids = encoded_batch.input_ids
            attention_mask = encoded_batch.attention_mask
            token_type_ids = encoded_batch.token_type_ids

            temp_emb = model([input_ids, attention_mask, token_type_ids])
            embeddings_lst.append(temp_emb)

        embeddings = tf.concat(embeddings_lst, axis=0)

        return embeddings

    return embed_f


def create_embed_f(old_embed_f, dataset_train, limit_num_sents, type: str, visualize=False):
    """Fine-tunes embeddings from USE or SBERT (using their embedding function). Returns new embed function."""

    split = Split(old_embed_f)
    X_train, y_train = split.get_X_y(dataset_train, limit_num_sents=limit_num_sents, set_type='train')

    emb_dim = X_train.shape[1]
    num_classes = len(set(np.asarray(y_train)))

    if type == 'softmax':
        model = ADBPretrainSoftmaxModel(emb_dim, num_classes)
    elif type == 'cosface':
        model = ADBPretrainCosFaceModel(emb_dim, num_classes)
    else:  # triplet_loss
        model = ADBPretrainTripletLossModel(emb_dim)

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

                X_train[idx:to_idx, :] = X_selection
                y_train[idx:to_idx] = y_selection

                num_seen_samples += len(y_selection)

            idx = to_idx

        X_train = tf.convert_to_tensor(X_train)
        y_train = tf.convert_to_tensor(y_train)

    model.compile(optimizer=optimizers.Adam(learning_rate=2e-5), loss=loss, metrics=['accuracy'])

    if type in ['softmax', 'triplet_loss']:
        X = X_train
    else:  # cosface
        X = [X_train, y_train]

    model.fit(X, y_train, epochs=40, shuffle=shuffle, batch_size=batch_size)

    if visualize:
        pca = PCA(n_components=2)

        # visualize original embeddings
        X_train_pca = pca.fit_transform(X_train)
        visualize_2d_data(X_train_pca, y_train, title=f'Embeddings')

        # visualize embeddings after pre-training
        embeddings_lst = []
        for batch in batches(X_train, 32):
            temp_emb = model(batch)
            embeddings_lst.append(temp_emb)

        embeddings = tf.concat(embeddings_lst, axis=0)

        embeddings_pca = pca.fit_transform(embeddings)
        visualize_2d_data(embeddings_pca, y_train, title=f'Embeddings after pre-training with {type}')

    def embed_f(X):
        embeddings_lst = []

        for batch in batches(X, 32):  # iterate in batches of size 32
            X = old_embed_f(batch)

            temp_emb = model(X)
            embeddings_lst.append(temp_emb)

        embeddings = tf.concat(embeddings_lst, axis=0)

        return embeddings

    return embed_f
