from utils import RESULTS_CLINC150_RIS_PATH

import os, json
import matplotlib.pyplot as plt
import numpy as np

# model_name = 'AdaptiveDecisionBoundaryNN_euclidean'
# name = 'ADB'

model_name = 'ADBThreshold'
name = 'Proposed method'

path = os.path.join(RESULTS_CLINC150_RIS_PATH, f'{model_name}_results.json')

with open(path, 'r') as f:
    results = json.load(f)

y_labels = np.linspace(0, 100, 11)

for num_intents in results:
    # one graph per num_intents

    labels = list(results[num_intents].keys())
    fig, ax = plt.subplots()
    plt.title(f'{name} – {num_intents} intents')

    for e, emb in enumerate(['use_tran_cosface', 'use_tran_triplet_loss']):
        accuracy_lst = [d[emb]['accuracy'] for d in results[num_intents].values()]
        recall_lst = [d[emb]['recall'] for d in results[num_intents].values()]

        if emb == 'use_tran_cosface':
            emb_name = 'USE-TRAN + LMCL'
        else:
            emb_name = 'USE-TRAN + Triplet Loss'

        accuracy_p, = ax.plot(labels, accuracy_lst, '-', label=f'Accuracy ({emb_name})')
        recall_p, = ax.plot(labels, recall_lst, '--', color=accuracy_p.get_color(), label=f'Recall ({emb_name})')
        ax.legend()

    ax.set(xlabel='Number of training sentences per intent', ylabel='Accuracy and recall [%]')
    ax.set_xticks(labels)
    ax.set_yticks(y_labels)

    plt.show()
    # plt.savefig(f'{num_intents}_intents.pdf')
