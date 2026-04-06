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
        num_connections = int(num_connections)
        row_dim = int(row_dim)
        col_dim = int(col_dim)
        if num_connections > row_dim * col_dim:
            raise ValueError(f"Error: num_connections must be less than {row_dim * col_dim}")
        unique_pairs = set()
        while len(unique_pairs) < num_connections:
            pair = (np.random.randint(row_dim), np.random.randint(col_dim))
            unique_pairs.add(pair)
        return list(unique_pairs)    
   
    def reservoir(self):
        lr, desired_sr, omega, gamma, N_res, sparsity_res, sparsity_input = self.ind
        connectivity_res = 1 - sparsity_res
        num_connections_Wres = int(N_res * N_res * connectivity_res)

        Wres = torch.zeros((int(N_res), int(N_res)), device=self.device)
        unique_pairs_Wres = self.generate_unique_pairs(num_connections_Wres, N_res, N_res)
        for i, (r, c) in enumerate(unique_pairs_Wres):
            Wres[r, c] = torch.empty(1, device=self.device).uniform_(-1, 1)
           
        connectivity_input = 1 - sparsity_input
        num_connections_Win = int(N_res * self.input_dim * connectivity_input)

        Win = torch.zeros((int(N_res), self.input_dim), device=self.device)
        unique_pairs_Win = self.generate_unique_pairs(num_connections_Win, N_res, self.input_dim)
        for i, (r, c) in enumerate(unique_pairs_Win):
            Win[r, c] = torch.empty(1, device=self.device).uniform_(-1, 1)
       
        return Wres, Win
   
    def ensure_data_quality(self, y_data, res_states_data):
        if torch.isnan(res_states_data).any():
            print(f'Total NaNs in reservoir states data replaced with zero.')
            res_states_data[torch.isnan(res_states_data)] = 0
        if torch.isinf(y_data).any():
            raise ValueError("Error: Target states (y_data) contain Inf values.")
        if torch.isinf(res_states_data).any():
            raise ValueError("Error: Reservoir states data contain Inf values.")
        if torch.isnan(y_data).any():
            raise ValueError("Error: Target states (y_data) contain NaN values.")

    def Res_States(self, input_signal, initial_res_state, Wres, Win):
        try:
            lr, desired_sr, omega, gamma, N_res, _, _ = self.ind
   
            eigvals = torch.linalg.eigvals(Wres)
            sr = torch.max(torch.abs(eigvals))
            Wres = (Wres / sr) * desired_sr
   
            Win = Win / torch.norm(Win)
   
            reservoir_state = initial_res_state.to(self.device)
            internal_res = []
   
            for x in range(len(input_signal)):
                input_data = torch.tensor(input_signal.iloc[x].values, dtype=torch.float32, device=self.device)
   
                input_info = Win @ input_data
                reservoir_info = Wres @ reservoir_state
   
                zeta = torch.rand(int(N_res), device=self.device) * 2 - 1
                z = zeta / torch.norm(zeta)
   
                state_update = lr * reservoir_state + (1 - lr) * torch.tanh(gamma * input_info + reservoir_info + omega * z)
                reservoir_state = state_update
                internal_res.append(reservoir_state.unsqueeze(0))
   
            return torch.cat(internal_res, dim=0)
   
        except IndexError:
            print('--- IndexError occurred in reservoir ---')
            if len(input_signal) == 0:
                return None
       
    def ESN(self):
        Wres, Win = self.reservoir()
        train_size = self.train_size

        X_train = self.data.iloc[:(train_size - 1), :]
        Y_train = self.data.iloc[1:(train_size), 0]
        X_test = self.data.iloc[(train_size - 1):-1, :].reset_index(drop=True)
        Y_test = self.data.iloc[(train_size):, 0].reset_index(drop=True)

        alpha_space = [1e-11, 1e-6, 1e-3, 0.01, 0.1, 0.5, 1, 5, 10, 100]
        params_for_grid = {'alpha': alpha_space}

        grid_search = GridSearchCV(
            estimator=Ridge(),
            param_grid=params_for_grid,
            scoring='r2',
            cv=5,
            n_jobs=-1,
            error_score='raise',
            refit=False,
            verbose=1
        )
       
        starting_state = torch.zeros(int(self.ind[4]), device=self.device)
        res_states_train = self.Res_States(X_train, starting_state, Wres, Win)
        y_train_tensor = torch.tensor(Y_train.values, dtype=torch.float32, device=self.device)

        self.ensure_data_quality(y_train_tensor, res_states_train)

        initial_washout = 200
        X_train_ridge = res_states_train[initial_washout:].cpu().numpy()
        Y_train_ridge = y_train_tensor[initial_washout:].cpu().numpy()
       
        grid_search.fit(X_train_ridge, Y_train_ridge)

        ridge = Ridge(grid_search.best_params_['alpha'])
        ridge.fit(X_train_ridge, Y_train_ridge)
       
        previous_state = res_states_train[-1, :].clone()    
        preds = []
        reservoir_states_test = []

        for k in range(len(X_test)):
            input_k = X_test.iloc[[k]]
            state_test = self.Res_States(input_k, previous_state, Wres, Win)

            if torch.isnan(state_test).any():
                print(f'Total NaNs in reservoir states data replaced with zero.')
                state_test[torch.isnan(state_test)] = 0
            if torch.isinf(state_test).any():
                print("Warning: Inf values detected in state_test. Replacing with zero.")
                state_test[torch.isinf(state_test)] = 0

            pred = ridge.predict(state_test.cpu().numpy())
            preds.append(pred[0])

            previous_state = state_test[-1, :].clone()
            reservoir_states_test.append(state_test)
       
        preds_df = pd.DataFrame(preds)
        reservoir_states_test = torch.cat(reservoir_states_test, dim=0).cpu().numpy()
       
        mae = mean_absolute_error(Y_test, preds_df)
        mse = mean_squared_error(Y_test, preds_df)
        rmse = np.sqrt(mse)
        r2 = r2_score(Y_test, preds_df)
   
        true_diff = np.sign(np.diff(Y_test))
        pred_diff = np.sign(np.diff(preds))
        da = np.mean(true_diff == pred_diff)

        true_dir = (true_diff > 0).astype(int)
        pred_dir = (pred_diff > 0).astype(int)

        precision = precision_score(true_dir, pred_dir, zero_division=0)
        recall = recall_score(true_dir, pred_dir, zero_division=0)
        f1 = f1_score(true_dir, pred_dir, zero_division=0)
       
        print(f"R²:   {r2:.9f}")
        print(f"MAE:  {mae:.9f}")
        print(f"Directional Accuracy: {da:.2%}")
        print(f"Precision (direction): {precision:.2%}")
        print(f"Recall (direction):    {recall:.2%}")
        print(f"F1 Score (direction):  {f1:.2%}")
       
        return rmse, reservoir_states_test, X_test, Y_test, preds_df


    model.ESN()
