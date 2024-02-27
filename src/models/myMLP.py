import random
import time
from os.path import exists

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm.notebook import trange, tqdm

from .modelwrapper import ModelWrapper

def normalize_data(data):
    scaler = StandardScaler()
    scaler.fit(data)
    return pd.DataFrame(scaler.transform(data))


# create dataset
class CustomDataset(Dataset):
    def __init__(self, features, labels=None, transform=None, target_transform=None):
        self.features = features
        self.labels = labels

        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        features = self.features.iloc[idx, :]
        if self.transform:
            features = self.transform(features)

        if self.labels is not None:
            label = self.labels.iloc[idx]

            if self.target_transform:
                label = self.target_transform(label)

            return torch.tensor(features), torch.tensor(label, dtype=torch.long)
        else:
            return torch.tensor(features, dtype=torch.float64)


class MLP(nn.Module):
    def __init__(self, n_classes, n_features):
        super(MLP, self).__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(n_features, n_features // 2, dtype=torch.float64),
            nn.ReLU(),
            nn.Linear(n_features // 2, n_classes, dtype=torch.float64),
        )

    def forward(self, x):
        x = self.linear_relu_stack(x)
        return F.softmax(x, dim=1)


def calculate_accuracy(y_pred, y):
    top_pred = y_pred.argmax(1, keepdim=True)
    correct = top_pred.eq(y.view_as(top_pred)).sum()
    acc = correct.float() / y.shape[0]
    return acc


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


class MyMLP(ModelWrapper):
    def __init__(self, n_classes, n_features, dataset="beans"):
        SEED = 1234

        random.seed(SEED)
        np.random.seed(SEED)
        torch.manual_seed(SEED)
        torch.cuda.manual_seed(SEED)
        torch.backends.cudnn.deterministic = True

        model = MLP(n_classes, n_features)
        super().__init__(model=model, type="MLP")

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.003)
        criterion = nn.CrossEntropyLoss()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.criterion = criterion.to(self.device)
        self.loss = None
        if dataset=="beans":
            self.param_path = './models/meta_parameters/model_weights_beans.pth'
        else:
            self.param_path = './models/meta_parameters/model_weights.pth'

    def train_epoch(self, iterator):
        epoch_loss = 0
        epoch_acc = 0
        self.model.train()

        for (x, y) in tqdm(iterator, desc="Training", leave=False):
            x = x.to(self.device)
            y = y.to(self.device)

            self.optimizer.zero_grad()

            y_pred = self.model(x)

            loss = self.criterion(y_pred, y)

            acc = calculate_accuracy(y_pred, y)

            loss.backward()

            self.optimizer.step()

            epoch_loss += loss.item()
            epoch_acc += acc.item()

        return epoch_loss / len(iterator), epoch_acc / len(iterator)

    def evaluate(self, iterator):
        epoch_loss = 0
        epoch_acc = 0
        self.model.eval()

        with torch.no_grad():
            results = []

            for (x,y) in tqdm(iterator, desc="Evaluating", leave=False):
                x = x.to(self.device)
                y = y.to(self.device)

                y_pred = self.model(x)

                results.append(y_pred)

                loss = self.criterion(y_pred, y)

                acc = calculate_accuracy(y_pred, y)

                epoch_loss += loss.item()
                epoch_acc += acc.item()

        return epoch_loss / len(iterator), epoch_acc / len(iterator), results


    def fit(self, x_train, y_train):
        train_dataset = CustomDataset(features=normalize_data(x_train), labels=y_train)
        train_iterator = DataLoader(dataset=train_dataset, batch_size=128, shuffle=True)

        self.classes_ = np.array(list(set(y_train)))
        self.sklearn_is_fitted = True

        if not exists(self.param_path):
            print("Warning: MLP not trained jet, use fitNN instead")

        else:
            print("MLP: load parameters....")
            self.model.load_state_dict(torch.load(self.param_path))


    def fitNN(self, x_train, y_train, x_test, y_test):
        train_dataset = CustomDataset(features=normalize_data(x_train), labels=y_train)
        train_iterator = DataLoader(dataset=train_dataset, batch_size=128, shuffle=True)

        test_dataset = CustomDataset(features=normalize_data(x_test), labels=y_test)
        test_iterator = DataLoader(dataset=test_dataset, batch_size=128, shuffle=False)

        self.classes_ = np.array(list(set(y_train)))
        self.sklearn_is_fitted = True

        if not exists(self.param_path):
            print("MLP: train....")

            EPOCHS = 100
            best_test_loss = np.inf

            for epoch in trange(EPOCHS):
                train_loss, train_acc = self.train_epoch(train_iterator)
                test_loss, test_acc, _ = self.evaluate(test_iterator)

                print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc * 100:.2f}%')
                print(f'\tValid Loss: {test_loss:.3f} | Valid Acc: {test_acc * 100:.2f}%')
                if test_loss < best_test_loss:
                    print("Save Model")
                    best_test_loss = test_loss
                    torch.save(self.model.state_dict(), self.param_path)

        else:
            print("MLP: load parameters....")
            self.model.load_state_dict(torch.load(self.param_path))

    def predict_acc(self, x_test, y_test):
        test_dataset = CustomDataset(features=normalize_data(x_test), labels=y_test)
        test_iterator = DataLoader(dataset=test_dataset, batch_size=128, shuffle=False)

        loss, acc, scores = self.evaluate(test_iterator)
        scores = [i[0] for i in scores]

        self.acc = acc
        self.loss = loss
        return acc

    def predict(self, x_test):
        return [np.argmax(i) for i in self.predict_proba(x_test)]

    def predict_proba(self, x_test):
        test_dataset = CustomDataset(features=normalize_data(x_test), labels=None)
        test_iterator = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False)

        self.model.eval()
        with torch.no_grad():
            results = []

            for x in tqdm(test_iterator, desc="Evaluating", leave=False):
                x = x.to(self.device)
                y_pred = self.model(x)
                results.append(y_pred)
        # convert to 2d nparray
        return np.array([i[0].numpy() for i in results])
