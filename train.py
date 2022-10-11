import pickle
import torch
from model import *
import torch.utils.data as Data
from torch import optim
from sklearn.preprocessing import train_test_split
import numpy as np
from sklearn.metrics import accuracy_score

class someDataset(Data.Dataset):
    def __init__(self, data, label):
        self.data = data
        self.label = label
    def __len__(self):
        return len(self.data)
    def __getitem__(self, index):
        return self.data[index], self.label[index]

def initialize(lr, weight_decay):
    model = CNN_baseline()
    optimizer = optim.Adam(model.parameters(), lr = lr, weight_decay= weight_decay)
    criterion = nn.CrossEntropyLoss()
    return model, optimizer, criterion

def train(data, label, model, loss_fcn, optimizer, epoch, batch_size, lr=0.0001, weight_decay=1e-5):
    model, optimizer, criterion = initialize(lr, weight_decay)
    X_train, X_val, y_train, y_val = train_test_split(data, label, test_size=0.3, random_state=42)
    train_loader = Data.DataLoader(someDataset(X_train, y_train), shuffle=True, batch_size=batch_size, drop_last=True)
    val_loader = Data.DataLoader(someDataset(X_val, y_val), shuffle=False, batch_size=batch_size, drop_last=True)
    for e in range(epoch):
        train_loss, val_loss = [], []
        train_acc, val_acc = [], []
        for i, batch in enumerate(train_loader, 0):
            optimizer.zero_grad()
            train_input, target = batch
            predict = model(train_input)
            loss = criterion(predict, target)
            loss.backward()
            optimizer.step()
            train_acc.append(accuracy_score(target, torch.argmax(predict, dim=1)))
            train_loss.append(loss.item())
        with torch.no_grad():
            for i, batch in enumerate(val_loader, 0):
                val_input, target = batch
                predict = model(val_input)
                loss = criterion(predict, target)
                val_acc.append(accuracy_score(target, torch.argmax(predict, dim=1)))
                val_loss.append(loss.item())
        print("Epoch [{}/{}], training loss:{:.5f}, validation loss:{:.5f}, train F1: {:.5f}, validation F1: {:.5f}, train acc: {:.2f}, valid acc: {:.2f}".format(e + 1, epoch, np.mean(train_loss),np.mean(val_loss),np.mean(train_acc), np.mean(val_acc)))

    return