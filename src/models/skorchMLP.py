from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import torch
import torch.nn as nn
import torch.nn.functional as F

from skorch import NeuralNetClassifier


class MyModule(nn.Module):
    def __init__(self, n_classes, n_features):
        super().__init__()

        self.linear_relu_stack = nn.Sequential(
            nn.Linear(n_features, n_features // 2, dtype=torch.float64),
            nn.ReLU(),
            nn.Linear(n_features // 2, n_classes, dtype=torch.float64),
        )

    def forward(self, x, **kwargs):
        x = self.linear_relu_stack(x)
        return F.softmax(x, dim=1)


def skorchMLP(n_classes, n_features):
    mlp_model = NeuralNetClassifier(
        MyModule(n_classes, n_features),
        max_epochs=10,
        criterion=nn.CrossEntropyLoss(),
        optimizer=torch.optim.Adam,
        lr=0.1,
        batch_size=10,
        # Shuffle training data on each epoch
        iterator_train__shuffle=True,
    )
    pipe = Pipeline([
        ('scale', StandardScaler()),
        ('net', mlp_model),
    ])
    return pipe

#skorchMLP_model = skorchMLP(n_classes=len(set(y_train)), n_features=X_train.shape[1])
#skorchMLP_model.fit(X_train.astype(np.double), torch.tensor(y_train.to_numpy(), dtype=torch.long))
#y_pred = skorchMLP_model.predict(X_test.astype(np.double))
#print("skorchMLP acc: ", (y_pred == y_test).mean())