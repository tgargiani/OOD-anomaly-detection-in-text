from utils import Split
from custom_models import ADBPretrainSoftmaxModel

import numpy as np
from transformers import AutoTokenizer
from tensorflow.keras import losses, optimizers


def create_bert_softmax_embed_f(dataset_train, limit_num_sents):
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

    model = ADBPretrainSoftmaxModel(SEQ_LEN, num_classes)
    model.compile(optimizer=optimizers.Adam(learning_rate=2e-5), loss=losses.SparseCategoricalCrossentropy(),
                  metrics=['accuracy'])
    model.fit([train_input_ids, train_attention_mask, train_token_type_ids], y_train, epochs=10)

    def embed_f(X):
        encoded_batch = tokenizer(X, padding='max_length', max_length=SEQ_LEN, truncation=True,
                                  return_attention_mask=True, return_tensors='tf')
        input_ids = encoded_batch.input_ids
        attention_mask = encoded_batch.attention_mask
        token_type_ids = encoded_batch.token_type_ids

        embeddings = model([input_ids, attention_mask, token_type_ids])

        return embeddings

    return embed_f
