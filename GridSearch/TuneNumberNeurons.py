#import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
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

class Classifier(nn.Module):
    def __init__(self, n_neurons=12):
        super().__init__()
        self.layer = nn.Linear(cols, n_neurons)
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(0.1)
        self.output = nn.Linear(n_neurons, 1)
        self.prob = nn.Sigmoid()
        self.weight_constraint = 2.0
        # manually init weights
        init.kaiming_uniform_(self.layer.weight)
        init.kaiming_uniform_(self.output.weight)

    def forward(self, x):
        # maxnorm weight before actual forward pass
        with torch.no_grad():
            norm = self.layer.weight.norm(2, dim=0, keepdim=True).clamp(min=self.weight_constraint / 2)
            desired = torch.clamp(norm, max=self.weight_constraint)
            self.layer.weight *= (desired / norm)
        # actual forward pass
        x = self.act(self.layer(x))
        x = self.dropout(x)
        x = self.prob(self.output(x))
        return x

# create model with skorch
model = NeuralNetClassifier(
    Classifier,
    criterion=nn.BCELoss,
    optimizer=optim.Adamax,
    max_epochs=100,
    batch_size=10,
    verbose=False
)

# define the grid search parameters
param_grid = {
    'module__n_neurons': [1, 5, 10, 15, 20, 25, 30]
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
