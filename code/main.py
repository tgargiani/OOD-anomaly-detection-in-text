from utils import DS_CLINC150_PATH, USE_DAN_PATH, USE_TRAN_PATH, print_results, get_intents_selection, get_filtered_lst, \
    save_results, DS_ROSTD_PATH, DS_LL_PATH, cross_val_evaluate
from custom_embeddings import create_bert_embed_f, create_embed_f
from CosineSimilarity import CosineSimilarity
from NeuralNets import BaselineNN, BaselineNNExtraLayer, CosFaceNN, CosFaceNNExtraLayer, CosFaceLOFNN, ArcFaceNN, \
    ArcFaceNNExtraLayer, AdaptiveDecisionBoundaryNN
from ADBThreshold import ADBThreshold

import os, json, copy
import tensorflow_hub as hub
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from statistics import mean

RANDOM_SELECTION = False  # am I testing using the random selection of IN intents?
repetitions = 5  # number of evaluations when using random selection
LIMIT_NUM_SENTS = None  # either None (i.e. no limit) or int with value > 0 (i.e. maximal number of sentences per class).
# LIMIT_NUM_SENTS is ignored when RANDOM_SELECTION is True

imports = []

# ------------------------------------------------------------
# from ood_train import evaluate
#
# imports.append((evaluate, [
#     ADBThreshold(alpha=1.45),  # best for CLINC150
#     ADBThreshold(alpha=0.2),  # best for ROSTD
#     ADBThreshold(alpha=1.0), # best for Lucid Lindia
#     AdaptiveDecisionBoundaryNN('angular'),
#     AdaptiveDecisionBoundaryNN('cosine'),
#     AdaptiveDecisionBoundaryNN('euclidean'),
#     ArcFaceNN(),
#     ArcFaceNNExtraLayer(),
#     CosFaceLOFNN(),
#     CosFaceNN(),
#     CosFaceNNExtraLayer(),
#     BaselineNN(),
#     BaselineNNExtraLayer(),
#     CosineSimilarity(),
#     LogisticRegression()]))
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
# dataset_name = 'rostd_full'
# dataset_path = os.path.join(DS_ROSTD_PATH, 'rostd_full.json')
# ------------------------------------------------------------
# dataset_name = 'lucid_lindia'
# categories = ['animals', 'books', 'brainquist', 'education', 'fashion', 'foods', 'habits', 'movies', 'music', 'science',
#               'smalltalk', 'sports', 'traveling']  # all
# ------------------------------------------------------------


if dataset_name != 'lucid_lindia':
    with open(dataset_path) as f:
        dataset = json.load(f)

time_pretraining = None  # keeps track of pre-training time in seconds (if there's ADB pre-training)

embedding_functions = {}  # uncomment them one by one when measuring memory usage or pre-training time
# embedding_functions['use_dan'] = hub.load(USE_DAN_PATH)
# embedding_functions['use_tran'] = hub.load(USE_TRAN_PATH)
# embedding_functions['sbert'] = SentenceTransformer('stsb-roberta-base').encode

# TO BE USED ONLY WITH ADAPTIVE DECISION BOUNDARY:
# embedding_functions['bert_softmax'], time_pretraining = create_bert_embed_f(dataset, LIMIT_NUM_SENTS, type='softmax')
# embedding_functions['bert_cosface'], time_pretraining = create_bert_embed_f(dataset, LIMIT_NUM_SENTS, type='cosface')
# embedding_functions['bert_triplet_loss'], time_pretraining = create_bert_embed_f(dataset, LIMIT_NUM_SENTS,
#                                                                                  type='triplet_loss')
#
# use_dan = hub.load(USE_DAN_PATH)
# embedding_functions['use_dan_softmax'], time_pretraining = create_embed_f(use_dan, dataset, LIMIT_NUM_SENTS,
#                                                                           type='softmax')
# embedding_functions['use_dan_cosface'], time_pretraining = create_embed_f(use_dan, dataset, LIMIT_NUM_SENTS,
#                                                                           type='cosface', visualize=False)
# embedding_functions['use_dan_triplet_loss'], time_pretraining = create_embed_f(use_dan, dataset, LIMIT_NUM_SENTS,
#                                                                                type='triplet_loss', visualize=False,
#                                                                                emb_name='use_dan')
#
# use_tran = hub.load(USE_TRAN_PATH)
# embedding_functions['use_tran_softmax'], time_pretraining = create_embed_f(use_tran, dataset, LIMIT_NUM_SENTS,
#                                                                            type='softmax')
# embedding_functions['use_tran_cosface'], time_pretraining = create_embed_f(use_tran, dataset, LIMIT_NUM_SENTS,
#                                                                            type='cosface')
# embedding_functions['use_tran_triplet_loss'], time_pretraining = create_embed_f(use_tran, dataset, LIMIT_NUM_SENTS,
#                                                                                 type='triplet_loss', visualize=False,
#                                                                                 emb_name='use_tran')
#
# sbert = SentenceTransformer('stsb-roberta-base').encode
# embedding_functions['sbert_softmax'], time_pretraining = create_embed_f(sbert, dataset, LIMIT_NUM_SENTS, type='softmax')
# embedding_functions['sbert_cosface'], time_pretraining = create_embed_f(sbert, dataset, LIMIT_NUM_SENTS, type='cosface')
# embedding_functions['sbert_triplet_loss'], time_pretraining = create_embed_f(sbert, dataset, LIMIT_NUM_SENTS,
#                                                                              type='triplet_loss', emb_name='sbert')
#
# embedding_functions[
#     'placeholder'] = None  # use this placeholder if you want to use ADB pre-training on lucid_lindia dataset.
# # Choose pre-training type in cross_val_evaluate.

if not RANDOM_SELECTION:
    for i in imports:
        evaluate = i[0]

        for emb_name, embed_f in embedding_functions.items():
            for model in i[1]:
                model_name = type(model).__name__

                if dataset_name != 'lucid_lindia':
                    results_dct = evaluate(dataset, model, model_name, embed_f, LIMIT_NUM_SENTS)

                    if time_pretraining is not None:
                        results_dct['time_pretraining'] = time_pretraining
                else:
                    results_dct, emb_name = cross_val_evaluate(categories, evaluate, model, model_name, emb_name,
                                                               embed_f, LIMIT_NUM_SENTS)

                print_results(dataset_name, model_name, emb_name, results_dct)
else:
    # use only CLINC150 dataset
    # use only ADB with euclidean distance and USE-TRAN embeddings with CosFace and Triplet Loss pre-training for random selection
    from ood_train import evaluate

    # model = AdaptiveDecisionBoundaryNN('euclidean')
    model = ADBThreshold(1.45)  # best for CLINC150
    model_name = type(model).__name__
    use_tran = hub.load(USE_TRAN_PATH)

    results_dct = {}
    # STRUCTURE:
    # legend: dict -> key
    #
    # results_dct -> num_intents -> limit_num_sents -> use_tran_cosface OR use_tran_triplet_loss ->
    # -> accuracy, float AND recall, float AND far, float AND frr, float AND accuracy_lst, list AND
    # AND recall_lst, list AND far_lst, list AND frr_lst, list

    for num_intents in [2, 3, 4, 5, 6, 8, 10, 15, 20, 25, 50, 100, 150]:  # limit number of intents
        results_dct[num_intents] = {}

        for limit_num_sents in range(4, 101, 4):  # 4,8,..,100
            results_dct[num_intents][limit_num_sents] = {}

            dct_shortcut = results_dct[num_intents][limit_num_sents]
            dct_shortcut['use_tran_cosface'] = {'accuracy': None, 'recall': None, 'far': None, 'frr': None,
                                                'accuracy_lst': [], 'recall_lst': [], 'far_lst': [], 'frr_lst': []}
            dct_shortcut['use_tran_triplet_loss'] = {'accuracy': None, 'recall': None, 'far': None, 'frr': None,
                                                     'accuracy_lst': [], 'recall_lst': [], 'far_lst': [], 'frr_lst': []}

            for i in range(repetitions):
                selection = get_intents_selection(dataset['train'],
                                                  num_intents)  # selected intent labels: (num_samples, ) np.ndarray

                filt_train = get_filtered_lst(dataset['train'],
                                              selection)  # almost the same as int_ds['train'] but filtered according to selection
                # filt_val = get_filtered_lst(dataset['val'], selection)  # ADB doesn't need validation split
                filt_test = get_filtered_lst(dataset['test'], selection)

                modified_dataset = copy.deepcopy(
                    dataset)  # deepcopy in order to not modify the original dict
                modified_dataset['train'] = filt_train
                # modified_dataset['val'] = filt_val  # ADB doesn't need validation split
                modified_dataset['test'] = filt_test

                embedding_functions = {}
                embedding_functions['use_tran_cosface'], _ = create_embed_f(use_tran, modified_dataset, limit_num_sents,
                                                                            type='cosface', emb_name='use_tran',
                                                                            visualize=False)
                embedding_functions['use_tran_triplet_loss'], _ = create_embed_f(use_tran, modified_dataset,
                                                                                 limit_num_sents, type='triplet_loss',
                                                                                 emb_name='use_tran', visualize=False)

                for emb_name, embed_f in embedding_functions.items():
                    temp_res = evaluate(modified_dataset, model, model_name, embed_f,
                                        limit_num_sents)  # temporary results

                    dct_shortcut[emb_name]['accuracy_lst'].append(temp_res['accuracy'])
                    dct_shortcut[emb_name]['recall_lst'].append(temp_res['recall'])
                    dct_shortcut[emb_name]['far_lst'].append(temp_res['far'])
                    dct_shortcut[emb_name]['frr_lst'].append(temp_res['frr'])

            dct_cosface = results_dct[num_intents][limit_num_sents]['use_tran_cosface']
            dct_triplet_loss = results_dct[num_intents][limit_num_sents]['use_tran_triplet_loss']

            dct_cosface['accuracy'] = round(mean(dct_cosface['accuracy_lst']), 1)
            dct_cosface['recall'] = round(mean(dct_cosface['recall_lst']), 1)
            dct_cosface['far'] = round(mean(dct_cosface['far_lst']), 1)
            dct_cosface['frr'] = round(mean(dct_cosface['frr_lst']), 1)

            dct_triplet_loss['accuracy'] = round(mean(dct_triplet_loss['accuracy_lst']), 1)
            dct_triplet_loss['recall'] = round(mean(dct_triplet_loss['recall_lst']), 1)
            dct_triplet_loss['far'] = round(mean(dct_triplet_loss['far_lst']), 1)
            dct_triplet_loss['frr'] = round(mean(dct_triplet_loss['frr_lst']), 1)

            print(f'{repetitions} times random selection {num_intents} intents, {limit_num_sents} sentences.')
            print_results(dataset_name, model_name, 'use_tran_cosface', dct_cosface)
            print_results(dataset_name, model_name, 'use_tran_triplet_loss', dct_triplet_loss)

        save_results(results_dct, dataset_name, model_name)
