import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch_geometric.nn import VGAE
import matplotlib.pyplot as plt
from torch_geometric.data import Data
import torch.nn as nn
from torch_geometric.utils import from_scipy_sparse_matrix, train_test_split_edges, add_self_loops
from scipy.sparse import load_npz
from sklearn.preprocessing import StandardScaler
from model import GCNEncoder, train, test, VariationalGAE
import argparse


# Argument Parser
parser = argparse.ArgumentParser(description="Run VGAE on different versions of protein data")
parser.add_argument("--version", type=int, choices=[0, 1, 2, 3], required=True,
                    help="Select version: 0 for No Node Features, 1 for Combined Features, 2 for RNA Features, 3 for Protein Expression Features")
args = parser.parse_args()

# Load Data
ppi_df = pd.read_csv('../Data/9606.protein.links.v12.0.txt', delimiter=' ')
rna_pca_df = pd.read_csv('../Data/PPI_RNA_only.csv')
pe_pca_df = pd.read_csv('../Data/PPI_protein_only.csv')
combined_feature_df = pd.read_csv('../Data/PPI_RNA_Protein_combined.csv')

# Load Precomputed Adjacency Matrix
adj_matrix = load_npz('../Data/adj_matrix_scaled.npz')  # Shape: (9466, 9466)

# Select Feature Matrix Based on Version
if args.version == 0:
    print("Running Version 0: No Node Features (Only PPI Data)")
    num_nodes = adj_matrix.shape[0]
    feature_matrix = np.ones((num_nodes, 1))  # Set node features to ones
elif args.version == 1:
    print("Running Version 1: Combined Features (RNA + Protein Expression)")
    feature_matrix = combined_feature_df.values
elif args.version == 2:
    print("Running Version 2: RNA Features Only")
    feature_matrix = rna_pca_df.values
elif args.version == 3:
    print("Running Version 3: Protein Expression Features Only")
    feature_matrix = pe_pca_df.values

# Standardize Feature Matrix (Only if features exist)
if args.version > 0:
    feature_matrix = StandardScaler().fit_transform(feature_matrix)

# Convert to PyG Data Format
edge_index, edge_attr = from_scipy_sparse_matrix(adj_matrix)
data = Data(edge_index=edge_index, edge_attr=edge_attr, x=torch.tensor(feature_matrix, dtype=torch.float))

# Split Edges
data = train_test_split_edges(data, val_ratio=0.2, test_ratio=0.2)
data.train_pos_edge_index, _ = add_self_loops(data.train_pos_edge_index)

# Define Model
in_channels = data.x.shape[1]
out_channels = 32
model = VariationalGAE(GCNEncoder(in_channels, out_channels))
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=0.0005)

# Train Model and Pass Version Number
metrics_dict = train(model, data, optimizer, version=args.version)

# Convert Metrics to NumPy Arrays
auc_values = np.array(metrics_dict["AUC"])
ap_values = np.array(metrics_dict["AP"])
precision_values = np.array(metrics_dict["Precision"], dtype=object)
recall_values = np.array(metrics_dict["Recall"], dtype=object)

# Determine Filename
metrics_filename = f"metrics_version_{args.version}.npz"

# Save Metrics
np.savez_compressed(metrics_filename,
                    auc_values=auc_values,
                    ap_values=ap_values,
                    precision_values=precision_values,
                    recall_values=recall_values,
                    allow_pickle=True)

print(f"All metrics saved to {metrics_filename}")
