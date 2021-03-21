from utils import DS_CLINC150_PATH, USE_DAN_PATH, USE_TRAN_PATH, print_results
from custom_embeddings import create_bert_embed_f
from CosineSimilarity import CosineSimilarity
from NeuralNets import BaselineNN, BaselineNNExtraLayer, CosFaceNN, CosFaceNNExtraLayer, CosFaceLOFNN, ArcFaceNN, \
    ArcFaceNNExtraLayer, AdaptiveDecisionBoundaryNN

import os, json
import tensorflow_hub as hub
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression

LIMIT_NUM_SENTS = False

imports = []

# ------------------------------------------------------------
from ood_train import evaluate

imports.append((evaluate, [
    AdaptiveDecisionBoundaryNN(),
    ArcFaceNN(),
    ArcFaceNNExtraLayer(),
    CosFaceLOFNN(),
    CosFaceNN(),
    CosFaceNNExtraLayer(),
    BaselineNN(),
    BaselineNNExtraLayer(),
    CosineSimilarity(),
    LogisticRegression()]))
# ------------------------------------------------------------
# from ood_threshold import evaluate
#
# imports.append((evaluate, [
#     ArcFaceNN(),
#     ArcFaceNNExtraLayer(),
#     CosFaceNN(),
#     CosFaceNNExtraLayer(),
#     BaselineNN(),
#     BaselineNNExtraLayer(),
#     CosineSimilarity(),
#     LogisticRegression()]))
# ------------------------------------------------------------
# from ood_binary import evaluate
#
# imports.append((evaluate, [
#     ArcFaceNN(),
#     ArcFaceNNExtraLayer(),
#     CosFaceNN(),
#     CosFaceNNExtraLayer(),
#     BaselineNN(),
#     BaselineNNExtraLayer(),
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

embedding_functions = {}
embedding_functions['use_dan'] = hub.load(USE_DAN_PATH)
embedding_functions['use_tran'] = hub.load(USE_TRAN_PATH)
embedding_functions['sbert'] = SentenceTransformer('stsb-roberta-base').encode
# embedding_functions['bert_softmax'] = create_bert_embed_f(dataset['train'],
#                                                           LIMIT_NUM_SENTS, type='softmax')  # to be used only with ADB
# embedding_functions['bert_cosface'] = create_bert_embed_f(dataset['train'],
#                                                           LIMIT_NUM_SENTS, type='cosface')  # to be used only with ADB

for i in imports:
    evaluate = i[0]

    for emb_name, embed_f in embedding_functions.items():
        for model in i[1]:
            model_name = type(model).__name__
            results_dct = evaluate(dataset, model, model_name, embed_f, LIMIT_NUM_SENTS)

            print_results(dataset_name, model_name, emb_name, results_dct)
