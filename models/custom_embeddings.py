from utils import Split, batches
from custom_models import ADBPretrainSoftmaxModel, ADBPretrainCosFaceModel

import numpy as np
from transformers import AutoTokenizer
import tensorflow as tf
from tensorflow.keras import losses, optimizers


def create_bert_embed_f(dataset_train, limit_num_sents, type: str):
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
        model = ADBPretrainSoftmaxModel(SEQ_LEN, num_classes)
    else:  # cosface
        model = ADBPretrainCosFaceModel(SEQ_LEN, num_classes)

    model.compile(optimizer=optimizers.Adam(learning_rate=2e-5), loss=losses.SparseCategoricalCrossentropy(),
                  metrics=['accuracy'])

    if type == 'softmax':
        X = [train_input_ids, train_attention_mask, train_token_type_ids]
    else:  # cosface
        X = [train_input_ids, train_attention_mask, train_token_type_ids, y_train]

    model.fit(X, y_train, epochs=2)

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
