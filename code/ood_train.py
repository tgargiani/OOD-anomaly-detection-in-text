from utils import Split, NEEDS_VAL
from testing import Testing

import time, psutil


def evaluate(dataset, model, model_name, embed_f, limit_num_sents):
    split = Split(embed_f)

    # TRAINING
    start_time_train = time.time()

    # Split dataset
    if model_name == 'ADBThreshold' or model_name == 'AdaptiveDecisionBoundaryNN' or model_name == 'CosFaceLOFNN':
        train_dataset = dataset['train']
    else:
        train_dataset = dataset['train'] + dataset['oos_train']

    X_train, y_train = split.get_X_y(train_dataset, limit_num_sents=limit_num_sents,
                                     set_type='train')

    # Train
    if model_name in NEEDS_VAL:
        if model_name == 'CosFaceLOFNN':
            val_dataset = dataset['val']
        else:
            val_dataset = dataset['val'] + dataset['oos_val']

        X_val, y_val = split.get_X_y(val_dataset, limit_num_sents=limit_num_sents,
                                     set_type='val')

        model.fit(X_train, y_train, X_val, y_val)
    else:
        model.fit(X_train, y_train)

    end_time_train = time.time()

    # ------------------
    # from sklearn.decomposition import PCA
    # from utils import visualize_2d_data
    # pca = PCA(n_components=2)
    #
    # # visualize original embeddings
    # X_train_pca = pca.fit_transform(X_train)
    # train_centroids = pca.transform(model.centroids)
    # visualize_2d_data(X_train_pca, y_train, title=f'Train embeddings',
    #                   centroids=train_centroids, delta=model.delta)
    # -----------

    memory = psutil.Process().memory_full_info().uss / (1024 ** 2)  # in megabytes

    # TESTING
    start_time_inference = time.time()

    # Split dataset
    X_test, y_test = split.get_X_y(dataset['test'] + dataset['oos_test'], limit_num_sents=None, set_type='test')

    # ------------------
    # X_test_pca = pca.fit_transform(X_test)
    # test_centroids = pca.transform(model.centroids)
    # visualize_2d_data(X_test_pca, y_test, title=f'Test embeddings',
    #                   centroids=test_centroids, delta=model.delta)
    # print(model.delta)
    # ------------------

    if model_name == 'ADBThreshold' or model_name == 'AdaptiveDecisionBoundaryNN' or model_name == 'CosFaceLOFNN':
        model.oos_label = split.intents_dct['oos']

    # Test
    testing = Testing(model, X_test, y_test, model_name, split.intents_dct['oos'])
    results_dct = testing.test_train()

    end_time_inference = time.time()

    results_dct['time_train'] = round(end_time_train - start_time_train, 1)
    results_dct['time_inference'] = round(end_time_inference - start_time_inference, 1)
    results_dct['memory'] = round(memory, 1)

    return results_dct
