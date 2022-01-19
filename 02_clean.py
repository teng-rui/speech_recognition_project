# Transform data.
#
# Usage: python3 ./02_clean.py [--normalize] [--pca]


import argparse

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


parser = argparse.ArgumentParser()
parser.add_argument('--normalize', action='store_true')
parser.add_argument('--pca', action='store_true')
args = parser.parse_args()


with np.load('dataset_train.npz') as dataset:
    inputs_train = dataset['inputs']
    labels_train = dataset['labels']
with np.load('dataset_devel.npz') as dataset:
    inputs_devel = dataset['inputs']
    labels_devel = dataset['labels']
with np.load('dataset_test.npz') as dataset:
    inputs_test = dataset['inputs']
    labels_test = dataset['labels']


if args.normalize:
    print('Normalizing...')
    scaler = StandardScaler()
    inputs_train = scaler.fit_transform(inputs_train)
    inputs_devel = scaler.transform(inputs_devel)
    inputs_test = scaler.transform(inputs_test)

if args.pca:
    print('Performing dimensionality reduction...')
    pca = PCA(n_components=3300).fit(inputs_train)
    inputs_train = pca.transform(inputs_train)
    inputs_devel = pca.transform(inputs_devel)
    inputs_test = pca.transform(inputs_test)
    print(f'Variance retained: {np.sum(pca.explained_variance_ratio_) * 100:.2f}%')


print('Saving...')
np.savez_compressed(f'dataset_train_clean.npz', inputs=inputs_train, labels=labels_train)
np.savez_compressed(f'dataset_devel_clean.npz', inputs=inputs_devel, labels=labels_devel)
np.savez_compressed(f'dataset_test_clean.npz', inputs=inputs_test, labels=labels_test)
