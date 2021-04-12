from utils import RESULTS_CLINC150_RIS_PATH

import os, json
import matplotlib.pyplot as plt
import numpy as np

path = os.path.join(RESULTS_CLINC150_RIS_PATH, 'results.json')

with open(path, 'r') as f:
    results = json.load(f)

y_labels = np.linspace(0, 100, 11)

for num_intents in results:
    # one graph per num_intents

    labels = list(results[num_intents].keys())
    fig, ax = plt.subplots()
    plt.title(f'ADB – {num_intents} intents')

    for e, emb in enumerate(['use_tran_cosface', 'use_tran_triplet_loss']):
        accuracy_lst = [d[emb]['accuracy'] for d in results[num_intents].values()]
        recall_lst = [d[emb]['recall'] for d in results[num_intents].values()]

        accuracy_p, = ax.plot(labels, accuracy_lst, '-', label=f'{emb} – accuracy')
        recall_p, = ax.plot(labels, recall_lst, '--', color=accuracy_p.get_color(), label=f'{emb} – recall')
        ax.legend()

    ax.set(xlabel='Number of sentences per intent', ylabel='Accuracy and recall [%]')
    ax.set_xticks(labels)
    ax.set_yticks(y_labels)

    plt.show()
    # plt.savefig(f'{num_intents}_intents.pdf')
