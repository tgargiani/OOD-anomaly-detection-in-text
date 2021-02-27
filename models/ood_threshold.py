from utils import Split, find_best_threshold
from testing import Testing

import time
import numpy as np


def evaluate(dataset, model, embed_f, limit_num_sents: bool):
    model_name = type(model).__name__
    split = Split(embed_f)

    # TRAINING
    start_time_train = time.time()

    # Split dataset
    X_train, y_train = split.get_X_y(dataset['train'], limit_num_sents=limit_num_sents, set_type='train')
    X_val, y_val = split.get_X_y(dataset['val'] + dataset['oos_val'], limit_num_sents=limit_num_sents, set_type='val')

    # Train
    if model_name == 'NeuralNet' or model_name == 'NeuralNetExtraLayer':
        X_val_fit, y_val_fit = split.get_X_y(dataset['val'], limit_num_sents=limit_num_sents,
                                             set_type='val')  # validation split must contain same labels as train split

        model.fit(X_train, y_train, X_val_fit, y_val_fit)
    else:
        model.fit(X_train, y_train)

    # Find threshold
    val_predictions_labels = []  # used to find threshold
    pred_probs = model.predict_proba(X_val)  # function available in both scikit-learn and TF-Keras, returns numpy array

    pred_labels = np.argmax(pred_probs, axis=1)
    pred_similarities = np.take_along_axis(pred_probs, indices=np.expand_dims(pred_labels, axis=1), axis=1).squeeze()

    predictions = np.column_stack([pred_labels, pred_similarities])  # 2D list of [pred_label, similarity]

    for pred, true_label in zip(predictions, y_val):
        val_predictions_labels.append((pred, true_label))

    threshold = find_best_threshold(val_predictions_labels, split.intents_dct['oos'])

    end_time_train = time.time()

    # TESTING
    start_time_inference = time.time()

    # Split dataset
    X_test, y_test = split.get_X_y(dataset['test'] + dataset['oos_test'], limit_num_sents=limit_num_sents,
                                   set_type='test')

    # Test
    testing = Testing(model, X_test, y_test, model_name, split.intents_dct['oos'])
    results_dct = testing.test_threshold(threshold)

    end_time_inference = time.time()

    results_dct['time_train'] = round(end_time_train - start_time_train, 1)
    results_dct['time_inference'] = round(end_time_inference - start_time_inference, 1)
    results_dct['threshold'] = threshold  # store threshold value

    return results_dct
