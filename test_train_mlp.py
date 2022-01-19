import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import recall_score, confusion_matrix, ConfusionMatrixDisplay
import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader, Dataset


label_names = ['ARA', 'CHI', 'FRE', 'GER', 'HIN', 'ITA', 'JPN', 'KOR', 'SPA', 'TEL', 'TUR']

device = 'cuda'

torch.manual_seed(0)


class ImplDataset(Dataset):

    def __init__(self, name):
        super().__init__()
        with np.load(f'dataset_{name}_clean.npz') as dataset:
            self.inputs = dataset['inputs']
            self.labels = np.array([label_names.index(x) for x in dataset['labels']])

    def __len__(self):
        return self.inputs.shape[0]

    def __getitem__(self, index):
        return self.inputs[index, :], self.labels[index]


train_data = ImplDataset('train')
devel_data = ImplDataset('devel')
test_data = ImplDataset('test')

train_loader = DataLoader(train_data, batch_size=30, shuffle=True)
devel_loader = DataLoader(devel_data)
test_loader = DataLoader(test_data)


class ImplMLP(nn.Module):

    def __init__(self, input_size=6373, output_size=11):
        super().__init__()

        hidden1_size = 120
        hidden2_size = 60
        dropout = 0.5

        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden1_size),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(hidden1_size, hidden2_size),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(hidden2_size, output_size),
        )

    def forward(self, x):
        return self.layers(x)


model = ImplMLP()

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

E = 35


for epoch in range(E):

    total_loss = 0
    total_iter = 0

    model.to(device)
    model.train()

    for inputs, labels in train_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_iter += 1

    def predict(loader):
        model.to('cpu')
        model.eval()

        labels_all = []
        predicts_all = []

        with torch.no_grad():
            for inputs, labels in loader:
                outputs = model(inputs)
                predicts = outputs.argmax(dim=1)
                labels_all.extend(labels.tolist())
                predicts_all.extend(predicts.tolist())

        return labels_all, predicts_all

    def uar(loader):
        labels, predicts = predict(loader)
        return recall_score(labels, predicts, average='macro', zero_division=0)

    def plot_cm(loader):
        labels, predicts = predict(loader)
        disp_labels = label_names
        cm = confusion_matrix(labels, predicts)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=disp_labels)
        disp.plot()
    
    print(f'epoch {epoch + 1:>2}  train loss {total_loss / total_iter:.4f}  uar train {uar(train_loader):.4f} / devel {uar(devel_loader):.4f} / test {uar(test_loader):.4}')

    if epoch + 1 == E:
        plot_cm(test_loader)
        plt.savefig('mlp_cm.png')
