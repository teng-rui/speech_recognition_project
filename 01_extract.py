# Extract features for baseline model using openSMILE and store them in
# dataset_train.npz, dataset_devel.npz and dataset_test.npz.
# Each output file contains two arrays, namely inputs and labels.
#
# Usage: python3 ./01_extract.py


import os
from pathlib import Path

import numpy as np
import pandas as pd
from rich.progress import track

os.environ['PATH'] += os.pathsep + '/m/work/courses/T/S/89/5150/general/bin/'  # SoX dependency for openSMILE
import opensmile


root = Path('/m/work/courses/T/S/89/5150/general/data/native_language')


df0 = pd.read_csv(root / 'lab' / 'ComParE2016_Nativeness.tsv', sep='\t').drop(columns=['promptId'])
df1 = pd.read_csv(root / 'lab' / 'ComParE2016_Nativeness-test.tsv', sep='\t')

df_train = df0[df0.file_name.str.startswith('train')]  # training set
df_devel = df0[df0.file_name.str.startswith('devel')]  # validation set
df_test = df1                                          # test set


smile = opensmile.Smile(
    feature_set=opensmile.FeatureSet.ComParE_2016,
    feature_level=opensmile.FeatureLevel.Functionals,
)

dataset_list = [
    ('train', df_train),
    ('devel', df_devel),
    ('test', df_test),
]

for dataset_name, dataset_df in dataset_list:

    inputs = []
    for file_name in track(dataset_df.file_name, description=f'Processing {dataset_name} dataset...'):
        features = smile.process_file(root / 'wav' / file_name).to_numpy()
        inputs.append(features)
    inputs = np.concatenate(inputs, dtype='f4')

    labels = np.array(dataset_df.L1, dtype='U3')

    np.savez_compressed(f'dataset_{dataset_name}.npz', inputs=inputs, labels=labels)
