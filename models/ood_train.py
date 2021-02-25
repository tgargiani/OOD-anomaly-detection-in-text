from utils import Split
from testing import Testing

import time


def evaluate(dataset, model, embed_f, limit_num_sents: bool):
    model_name = type(model).__name__
    split = Split(embed_f)

    # TRAINING
    start_time_train = time.time()

    # Split dataset
    X_train, y_train = split.get_X_y(dataset['train'] + dataset['oos_train'], limit_num_sents=limit_num_sents,
                                     set_type='train')

    # Train
    if model_name == 'keras_something':
        pass  # gonna need a different fit
    else:
        model.fit(X_train, y_train)

    end_time_train = time.time()

    # TESTING
    start_time_inference = time.time()

    # Split dataset
    X_test, y_test = split.get_X_y(dataset['test'] + dataset['oos_test'], limit_num_sents=limit_num_sents,
                                   set_type='test')

    # Test
    testing = Testing(model, X_test, y_test, model_name, split.intents_dct['oos'])
    results_dct = testing.test_train()

    end_time_inference = time.time()

    results_dct['time_train'] = round(end_time_train - start_time_train, 1)
    results_dct['time_inference'] = round(end_time_inference - start_time_inference, 1)

    return results_dct
