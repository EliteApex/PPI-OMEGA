import torch
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import ParameterGrid
from torch_geometric.utils import train_test_split_edges, add_self_loops
from model import GCNEncoder, VariationalGAE, train
from torch_geometric.data import Data
from scipy.sparse import load_npz
import pandas as pd
from sklearn.preprocessing import StandardScaler
import argparse
import copy

def hyperparameter_search(data, param_grid, out_channels=32):
    best_auc = -np.inf
    best_params = None

    param_combinations = list(ParameterGrid(param_grid))
    
    print(f"Total hyperparameter combinations: {len(param_combinations)}")

    with tqdm(total=len(param_combinations), desc="Hyperparameter Search", unit="config") as pbar:
        for params in param_combinations:
            pbar.set_postfix(params=params)  # Update the progress bar with params info

            # Split data once
            data_split = copy.deepcopy(data)
            data_split = train_test_split_edges(data_split, val_ratio=0.2, test_ratio=0.2)
            data_split.train_pos_edge_index, _ = add_self_loops(data_split.train_pos_edge_index)

            # Define model and optimizer
            in_channels = data.x.shape[1]
            model = VariationalGAE(GCNEncoder(in_channels, out_channels))
            optimizer = torch.optim.Adam(model.parameters(), lr=params["lr"], weight_decay=params["weight_decay"])

            # Train model
            metrics_dict = train(model, data_split, optimizer, patience=40)
            mean_auc = max(metrics_dict["AUC"])  # Use best AUC from training

            # Update progress bar
            pbar.set_postfix(AUC=mean_auc, params=params)
            pbar.update(1)

            if mean_auc > best_auc:
                best_auc = mean_auc
                best_params = params

    print(f"\nBest Hyperparameters: {best_params} with AUC: {best_auc:.4f}")
    return best_params, best_auc


parser = argparse.ArgumentParser(description="Run hyperparameter search for VGAE")
args = parser.parse_args()

# Load Data
adj_matrix = load_npz("../Data/adj_matrix_scaled.npz")
combined_feature_df = pd.read_csv("../Data/PPI_RNA_Protein_combined.csv")
feature_matrix = StandardScaler().fit_transform(combined_feature_df.values)

# Convert to PyG Data Format
from torch_geometric.utils import from_scipy_sparse_matrix
edge_index, edge_attr = from_scipy_sparse_matrix(adj_matrix)
data = Data(edge_index=edge_index, edge_attr=edge_attr, x=torch.tensor(feature_matrix, dtype=torch.float))

# Define hyperparameter grid
param_grid = {
    "dropout": [0.3, 0.4, 0.5],
    "weight_decay": [5e-4, 1e-3, 5e-3],
    "lr": [0.001, 0.005, 0.01]
}

# Run hyperparameter search
best_params, best_auc = hyperparameter_search(data, param_grid)

# Save best parameters
best_params_df = pd.DataFrame([best_params])
best_params_df.to_csv("best_hyperparameters.csv", index=False)
print("Best hyperparameters saved to best_hyperparameters.csv")
