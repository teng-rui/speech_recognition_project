# Visualize confusion matrices.
#
# Usage: python3 ./04_visualize.py [--algo SVM/RF]


import argparse

from joblib import load
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


def plot_confusion_matrix(labels, predictions):
    disp_labels = np.array(['ARA', 'CHI', 'FRE', 'GER', 'HIN', 'ITA', 'JPN', 'KOR', 'SPA', 'TEL', 'TUR'], dtype='U3')
    cm = confusion_matrix(labels, predictions)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=disp_labels)
    disp.plot()


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--algo', default='SVM')
    args = parser.parse_args()

    with np.load('dataset_test_clean.npz') as dataset:
        inputs = dataset['inputs']
        labels = dataset['labels']

    if args.algo == 'SVM':
        svm = load('svm.joblib')
        predictions = svm.predict(inputs)
        plot_confusion_matrix(labels, predictions)
        plt.savefig('svm_cm.png')

    if args.algo == 'RF':
        rf = load('rf.joblib')
        predictions = rf.predict(inputs)
        plot_confusion_matrix(labels, predictions)
        plt.savefig('rf_cm.png')


if __name__ == '__main__':
    main()
