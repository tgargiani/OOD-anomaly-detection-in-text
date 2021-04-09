from utils import DS_CLINC150_PATH, USE_DAN_PATH, USE_TRAN_PATH, print_results
from custom_embeddings import create_bert_embed_f, create_embed_f
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
    AdaptiveDecisionBoundaryNN('angular'),
    AdaptiveDecisionBoundaryNN('cosine'),
    AdaptiveDecisionBoundaryNN('euclidean'),
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

with open(dataset_path) as f:
    dataset = json.load(f)

embedding_functions = {}  # uncomment them one by one when measuring memory usage
embedding_functions['use_dan'] = hub.load(USE_DAN_PATH)
# embedding_functions['use_tran'] = hub.load(USE_TRAN_PATH)
# embedding_functions['sbert'] = SentenceTransformer('stsb-roberta-base').encode

# TO BE USED ONLY WITH ADAPTIVE DECISION BOUNDARY:
# embedding_functions['bert_softmax'] = create_bert_embed_f(dataset['train'], LIMIT_NUM_SENTS, type='softmax')
# embedding_functions['bert_cosface'] = create_bert_embed_f(dataset['train'], LIMIT_NUM_SENTS, type='cosface')
# embedding_functions['bert_triplet_loss'] = create_bert_embed_f(dataset['train'], LIMIT_NUM_SENTS, type='triplet_loss')
#
# use_dan = hub.load(USE_DAN_PATH)
# embedding_functions['use_dan_softmax'] = create_embed_f(use_dan, dataset['train'], LIMIT_NUM_SENTS, type='softmax')
# embedding_functions['use_dan_cosface'] = create_embed_f(use_dan, dataset['train'], LIMIT_NUM_SENTS, type='cosface',
#                                                         visualize=False)
# embedding_functions['use_dan_triplet_loss'] = create_embed_f(use_dan, dataset['train'], LIMIT_NUM_SENTS,
#                                                              type='triplet_loss', visualize=False, emb_name='use_dan')
#
# use_tran = hub.load(USE_TRAN_PATH)
# embedding_functions['use_tran_softmax'] = create_embed_f(use_tran, dataset['train'], LIMIT_NUM_SENTS, type='softmax')
# embedding_functions['use_tran_cosface'] = create_embed_f(use_tran, dataset['train'], LIMIT_NUM_SENTS, type='cosface')
# embedding_functions['use_tran_triplet_loss'] = create_embed_f(use_tran, dataset['train'], LIMIT_NUM_SENTS,
#                                                               type='triplet_loss', visualize=False, emb_name='use_tran')
#
# sbert = SentenceTransformer('stsb-roberta-base').encode
# embedding_functions['sbert_softmax'] = create_embed_f(sbert, dataset['train'], LIMIT_NUM_SENTS, type='softmax')
# embedding_functions['sbert_cosface'] = create_embed_f(sbert, dataset['train'], LIMIT_NUM_SENTS, type='cosface')
# embedding_functions['sbert_triplet_loss'] = create_embed_f(sbert, dataset['train'], LIMIT_NUM_SENTS,
#                                                            type='triplet_loss', emb_name='sbert')

for i in imports:
    evaluate = i[0]

    for emb_name, embed_f in embedding_functions.items():
        for model in i[1]:
            model_name = type(model).__name__
            results_dct = evaluate(dataset, model, model_name, embed_f, LIMIT_NUM_SENTS)

            print_results(dataset_name, model_name, emb_name, results_dct)
