import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from skorch import NeuralNetClassifier
from sklearn.model_selection import GridSearchCV
from pandas.api.types import is_numeric_dtype
from sklearn.preprocessing import LabelEncoder
import sys
import pandas as pd


# load the dataset, split into input (X) and output (y) variables
if (len(sys.argv)==1):
    nameDataset= input ("Database name: ")
else:
    nameDataset = sys.argv[1]

dataset = pd.read_csv('dataset/'+nameDataset, delimiter=',')
cols = dataset.shape[1] - 1;
print(dataset.shape)
X = dataset.iloc[:,0:cols]
y = dataset.iloc[:,cols]

if (not is_numeric_dtype(y)):
    print("Preprocessing data class");
    encoder = LabelEncoder()
    encoder.fit(y)
    y = encoder.transform(y)

X = torch.tensor(X.values, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32).reshape(-1, 1)

# PyTorch classifier
class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.Linear(cols, 12)
        self.act = nn.ReLU()
        self.output = nn.Linear(12, 1)
        self.prob = nn.Sigmoid()

    def forward(self, x):
        x = self.act(self.layer(x))
        x = self.prob(self.output(x))
        return x

# create model with skorch
model = NeuralNetClassifier(
    Classifier,
    criterion=nn.BCELoss,
    max_epochs=500,
    batch_size=10,
    verbose=False
)

# define the grid search parameters
param_grid = {
    'optimizer': [optim.SGD, optim.RMSprop, optim.Adagrad, optim.Adadelta,
                  optim.Adam, optim.Adamax, optim.NAdam],
}
grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=3)
grid_result = grid.fit(X, y)

# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))
