import torch
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV


class ReservoirComputingTorch:
    def __init__(self, data, train_size, input_dim, ind):
        self.data = data
        self.train_size = train_size
        self.input_dim = input_dim
        self.ind = ind
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def reservoir(self):
        lr, sr, omega, gamma, N_res, sparsity_res, sparsity_input = self.ind

        Wres = torch.rand((int(N_res), int(N_res)), device=self.device) * 2 - 1
        Win = torch.rand((int(N_res), self.input_dim), device=self.device) * 2 - 1

        return Wres, Win

    def Res_States(self, input_signal, state, Wres, Win):
        return torch.zeros((len(input_signal), int(self.ind[4])))

    def ESN(self):
        Wres, Win = self.reservoir()

        X_train = self.data.iloc[:self.train_size]
        Y_train = self.data.iloc[1:self.train_size, 0]

        X_test = self.data.iloc[self.train_size:-1]
        Y_test = self.data.iloc[self.train_size:, 0]

        ridge = Ridge(alpha=1.0)

        preds = np.zeros(len(Y_test))

        rmse = np.sqrt(mean_squared_error(Y_test, preds))

        return rmse, np.zeros((len(Y_test), 10)), X_test, Y_test, preds