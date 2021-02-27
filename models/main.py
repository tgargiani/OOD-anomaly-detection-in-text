from utils import DS_CLINC150_PATH, USE_DAN_PATH, USE_TRAN_PATH, print_results
from CosineSimilarity import CosineSimilarity
from NeuralNets import NeuralNet, NeuralNetExtraLayer

import os, json
import tensorflow_hub as hub
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from tensorflow.keras import activations, losses

LIMIT_NUM_SENTS = False

embedding_functions = {'use_dan': hub.load(USE_DAN_PATH), 'use_tran': hub.load(USE_TRAN_PATH),
                       'sbert': SentenceTransformer('stsb-roberta-base').encode}
imports = []

# ------------------------------------------------------------
from ood_train import evaluate

imports.append((evaluate, [
    NeuralNetExtraLayer(activations.softmax, losses.SparseCategoricalCrossentropy()),
    NeuralNet(activations.softmax, losses.SparseCategoricalCrossentropy()),
    CosineSimilarity(),
    LogisticRegression()]))
# ------------------------------------------------------------
# from ood_threshold import evaluate
#
# imports.append((evaluate, [
#     NeuralNetExtraLayer(activations.softmax, losses.SparseCategoricalCrossentropy()),
#     NeuralNet(activations.softmax, losses.SparseCategoricalCrossentropy()),
#     CosineSimilarity(),
#     LogisticRegression()]))
# ------------------------------------------------------------
# from ood_binary import evaluate
#
# imports.append((evaluate, [
#     NeuralNetExtraLayer(activations.softmax, losses.SparseCategoricalCrossentropy()),
#     NeuralNet(activations.softmax, losses.SparseCategoricalCrossentropy()),
#     CosineSimilarity(),
#     LogisticRegression()]))
# ------------------------------------------------------------

# Load dataset
# ------------------------------------------------------------
dataset_name = 'clinc150-data_full'
dataset_path = os.path.join(DS_CLINC150_PATH, 'data_full.json')
# ------------------------------------------------------------
# dataset_name = 'clinc150-data_small'
# dataset_path = os.path.join(DS_CLINC150_PATH, 'data_small.json')
# ------------------------------------------------------------
# dataset_name = 'clinc150-binary_undersample'
# dataset_path = os.path.join(DS_CLINC150_PATH, 'binary_undersample.json')
# ------------------------------------------------------------

with open(dataset_path) as f:
    dataset = json.load(f)

for i in imports:
    evaluate = i[0]

    for emb_name, embed_f in embedding_functions.items():
        for model in i[1]:
            model_name = type(model).__name__
            results_dct = evaluate(dataset, model, embed_f, LIMIT_NUM_SENTS)

            print_results(dataset_name, model_name, emb_name, results_dct)
