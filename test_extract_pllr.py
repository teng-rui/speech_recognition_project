from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from scipy.io import wavfile
from sidekit.frontend.features import shifted_delta_cepstral
from sklearn.decomposition import PCA
import torch
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2ForCTC


root = Path('/m/work/courses/T/S/89/5150/general/data/native_language')


df0 = pd.read_csv(root / 'lab' / 'ComParE2016_Nativeness.tsv', sep='\t').drop(columns=['promptId'])
df1 = pd.read_csv(root / 'lab' / 'ComParE2016_Nativeness-test.tsv', sep='\t')

df_train = df0[df0.file_name.str.startswith('train')]  # training set
df_devel = df0[df0.file_name.str.startswith('devel')]  # validation set
df_test = df1                                          # test set


model_name = 'facebook/wav2vec2-large-xlsr-53'
device = 'cuda'


feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
model = Wav2Vec2ForCTC.from_pretrained(model_name).to(device)


def extract_pllr(data):

    # extract
    features = feature_extractor(data, sampling_rate=16000, padding=True, return_tensors='pt')
    input_values = features.input_values.to(torch.float32).to(device)
    attention_mask = features.attention_mask.to(device)
    with torch.no_grad():
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            logits = model(input_values, attention_mask=attention_mask).logits
        probs = torch.nn.functional.softmax(logits, dim=2)
    p = np.array(probs.cpu()).squeeze()
    pllr = np.log(p / (1 - p))

    # project
    n = pllr.shape[1]
    eye = np.identity(n, dtype=np.float32)
    one = np.ones(n, dtype=np.float32) / np.sqrt(n)
    proj = eye - np.outer(one, one)
    return np.einsum('kj,ij->ki', pllr, proj, dtype=np.float32)


dataset_list = [
    ('train', df_train),
    ('devel', df_devel),
    ('test', df_test),
]


is_train = True
pca = PCA(n_components=13)

for dataset_name, dataset_df in dataset_list:

    fs = []
    
    for file_name in dataset_df.file_name:
        print(file_name)
        _, file_data = wavfile.read(root / 'wav' / file_name, mmap=True)

        f = extract_pllr(file_data)  # pllr: (N, 32)
        fs.append(f)

    if is_train:
        fs_combined = np.vstack(fs)
        pca.fit(fs_combined)

    for i, f in enumerate(fs):

        f = pca.transform(f)  # pca: (N, 32) -> (N, 13)
        f = shifted_delta_cepstral(f, d=2, p=3, k=7)  # sdc: (N, 13) -> (N, 104)
        f = f.astype(np.float32)

        fs[i] = f

    np.savez_compressed(f'dataset_pllr_{dataset_name}.npz', *fs)

    is_train = False
