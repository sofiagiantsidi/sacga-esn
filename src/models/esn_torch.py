import torch
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, precision_score, recall_score, f1_score
from sklearn.model_selection import GridSearchCV


class ReservoirComputingTorch:
    def __init__(self, data, train_size, input_dim, ind):
        self.data = data
        self.train_size = train_size
        self.input_dim = input_dim
        self.ind = ind
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def generate_unique_pairs(self, num_connections, row_dim, col_dim):
        unique_pairs = set()
        while len(unique_pairs) < num_connections:
            pair = (np.random.randint(row_dim), np.random.randint(col_dim))
            unique_pairs.add(pair)
        return list(unique_pairs)

    def reservoir(self):
        lr, desired_sr, omega, gamma, N_res, sparsity_res, sparsity_input = self.ind

        Wres = torch.zeros((int(N_res), int(N_res)), device=self.device)
        Win = torch.zeros((int(N_res), self.input_dim), device=self.device)

        return Wres, Win

    def Res_States(self, input_signal, initial_res_state, Wres, Win):
        lr, desired_sr, omega, gamma, N_res, _, _ = self.ind

        reservoir_state = initial_res_state.to(self.device)
        internal_res = []

        for x in range(len(input_signal)):
            input_data = torch.tensor(input_signal.iloc[x].values, dtype=torch.float32, device=self.device)

            reservoir_state = reservoir_state + torch.tanh(input_data @ torch.randn_like(reservoir_state))

            internal_res.append(reservoir_state.unsqueeze(0))

        return torch.cat(internal_res, dim=0)

    def ESN(self):
        Wres, Win = self.reservoir()

        train_size = self.train_size

        X_train = self.data.iloc[:(train_size - 1), :]
        Y_train = self.data.iloc[1:train_size, 0]

        X_test = self.data.iloc[(train_size - 1):-1].reset_index(drop=True)
        Y_test = self.data.iloc[train_size:, 0].reset_index(drop=True)

        res_states_train = self.Res_States(
            X_train,
            torch.zeros(int(self.ind[4]), device=self.device),
            Wres,
            Win
        )

        X_train_ridge = res_states_train.cpu().numpy()
        Y_train_ridge = Y_train.values

        ridge = Ridge(alpha=1.0)
        ridge.fit(X_train_ridge, Y_train_ridge)

        preds = []
        reservoir_states_test = []

        previous_state = res_states_train[-1, :].clone()

        for k in range(len(X_test)):
            input_k = X_test.iloc[[k]]
            state_test = self.Res_States(input_k, previous_state, Wres, Win)

            pred = ridge.predict(state_test.cpu().numpy())
            preds.append(pred[0])

            previous_state = state_test[-1, :].clone()
            reservoir_states_test.append(state_test)

        preds_df = pd.DataFrame(preds)
        reservoir_states_test = torch.cat(reservoir_states_test, dim=0).cpu().numpy()

        rmse = np.sqrt(mean_squared_error(Y_test, preds_df))

        return rmse, reservoir_states_test, X_test, Y_test, preds_df
