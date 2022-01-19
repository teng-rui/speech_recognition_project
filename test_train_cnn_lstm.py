import numpy as np
import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from torch import optim
from torch.utils.data import Dataset, DataLoader


label_names = ['ARA', 'CHI', 'FRE', 'GER', 'HIN', 'ITA', 'JPN', 'KOR', 'SPA', 'TEL', 'TUR']

device = 'cuda'

torch.manual_seed(0)


class CustomDataset(Dataset):
    def __init__(self):
        with np.load('dataset_pllr_train.npz') as dataset:
            self.inputs = [dataset[name] for name in sorted(dataset)]
        with np.load('dataset_train.npz') as dataset:
            self.labels = [label_names.index(name) for name in dataset['labels']]
        self.count = len(self.inputs)
    
    def __len__(self):
        return self.count

    def __getitem__(self, index):
        return self.inputs[index], self.labels[index]


def collate(data):
    inputs, labels = zip(*data)
    inputs = [torch.tensor(i) for i in inputs]
    inputs = pad_sequence(inputs, batch_first=True)
    labels = torch.tensor(labels)
    return inputs.to(device), labels.to(device)


train_data = CustomDataset()
train_loader = DataLoader(train_data, batch_size=30, shuffle=True, collate_fn=collate)


class Net(nn.Module):
    def __init__(self, input_size=104, output_size=11):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(input_size, 200, kernel_size=5, padding='same'),
            nn.BatchNorm1d(200),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=4),
            nn.Conv1d(200, 300, kernel_size=5, padding='same'),
            nn.BatchNorm1d(300),
            nn.MaxPool1d(kernel_size=4),
            nn.ReLU(),

        )
        self.lstm = nn.LSTM(300, 50, batch_first=True)
        self.relu = nn.ReLU()
        self.linear = nn.Linear(50, output_size)

    def forward(self, x):
        n = x.size(0)
        x = x.permute(0, 2, 1)
        x = self.cnn(x)
        x = x.permute(0, 2, 1)
        _, (x, _) = self.lstm(x)
        x = x.view(n, -1)
        x = self.relu(x)
        x = self.linear(x)
        return x


net = Net()
net.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.001)

for epoch in range(10):

    total_loss = 0
    total_iter = 0

    for inputs, labels in train_loader:

        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_iter += 1

    print(f'epoch {epoch + 1}, loss {total_loss / total_iter}')

net.eval()
with torch.no_grad():
    for inputs, labels in train_loader:
        outputs = net(inputs)
        predicts = outputs.argmax(dim=1)
        print(labels)
        print(predicts)
