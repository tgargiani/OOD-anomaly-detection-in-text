from utils import Split, DS_CLINC150_PATH, NEEDS_VAL
from testing import Testing

import os, json, time, copy, psutil


def train_model_multi(model_multi, embed_f, limit_num_sents: bool):
    model_multi_name = type(model_multi).__name__

    # Load dataset
    dataset_multi_path = os.path.join(DS_CLINC150_PATH, 'data_full.json')

    with open(dataset_multi_path) as f:
        dataset_multi = json.load(f)

    dataset = dataset_multi

    # Split dataset
    split_multi = Split(embed_f)

    X_multi_train, y_multi_train = split_multi.get_X_y(dataset['train'], limit_num_sents=limit_num_sents,
                                                       set_type='train')

    start_time_inference_split = time.time()
    X_multi_test, y_multi_test = split_multi.get_X_y(dataset['test'] + dataset['oos_test'],
                                                     limit_num_sents=limit_num_sents,
                                                     set_type='test')
    time_inference_split = time.time() - start_time_inference_split

    # Train
    if model_multi_name in NEEDS_VAL:
        X_multi_val, y_multi_val = split_multi.get_X_y(dataset['val'], limit_num_sents=limit_num_sents, set_type='val')

        model_multi.fit(X_multi_train, y_multi_train, X_multi_val, y_multi_val)
    else:
        model_multi.fit(X_multi_train, y_multi_train)

    return model_multi, X_multi_test, y_multi_test, split_multi, time_inference_split


def evaluate(dataset, model, model_name, embed_f, limit_num_sents: bool):
    # TRAINING
    start_time_train = time.time()

    # Train multi-intent model
    model_multi = copy.copy(model)  # create a new independent instance of the model with the same parameters
    model_multi, X_multi_test, y_multi_test, split_multi, time_inference_split = train_model_multi(model_multi, embed_f,
                                                                                                   limit_num_sents)

    # Split dataset
    split = Split(embed_f)

    X_bin_train, y_bin_train = split.get_X_y(dataset['train'], limit_num_sents=False, set_type='train')

    # Train
    if model_name in NEEDS_VAL:
        X_bin_val, y_bin_val = split.get_X_y(dataset['val'], limit_num_sents=False, set_type='val')

        model.fit(X_bin_train, y_bin_train, X_bin_val, y_bin_val)
    else:
        model.fit(X_bin_train, y_bin_train)

    end_time_train = time.time()

    memory = psutil.Process().memory_full_info().uss / (1024 ** 2)  # in megabytes

    # TESTING
    start_time_inference = time.time()

    # Test
    testing = Testing(model_multi, X_multi_test, y_multi_test, model_name, split_multi.intents_dct['oos'],
                      bin_model=model, bin_oos_label=split.intents_dct['oos'])
    results_dct = testing.test_binary()

    end_time_inference = time.time()

    results_dct['time_train'] = round(end_time_train - start_time_train - time_inference_split, 1)
    results_dct['time_inference'] = round(end_time_inference - start_time_inference + time_inference_split, 1)
    results_dct['memory'] = round(memory, 1)

    return results_dct
