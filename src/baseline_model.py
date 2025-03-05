import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data
from torch_geometric.utils import from_scipy_sparse_matrix, train_test_split_edges, add_self_loops
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.preprocessing import StandardScaler
from scipy.sparse import load_npz

def main():
    # Load the precomputed (scaled) adjacency matrix.
    adj_matrix = load_npz('../Data/adj_matrix_scaled.npz')
    
    # Load the combined feature matrix.
    # (You can modify this to load RNA-only or Protein-only features if desired)
    combined_feature_df = pd.read_csv('../Data/PPI_RNA_Protein_combined.csv')
    feature_matrix = combined_feature_df.values
    feature_matrix = StandardScaler().fit_transform(feature_matrix)
    
    # Convert to PyG Data format.
    edge_index, edge_attr = from_scipy_sparse_matrix(adj_matrix)
    data = Data(edge_index=edge_index, edge_attr=edge_attr, x=torch.tensor(feature_matrix, dtype=torch.float))
    
    # Split the edges into training, validation, and test sets.
    data = train_test_split_edges(data, val_ratio=0.2, test_ratio=0.2)
    # training set only has positive samples, test set has half negative half positive
    data.train_pos_edge_index, _ = add_self_loops(data.train_pos_edge_index)
    
    # Construct ground truth for the test set:
    # Positive edges are labeled as 1, negative edges as 0.
    pos_count = data.test_pos_edge_index.size(1)
    neg_count = data.test_neg_edge_index.size(1)
    y_true = np.concatenate([np.ones(pos_count), np.zeros(neg_count)])
    
    # Determine the majority class in the test set.
    # (If counts are equal, the baseline will predict 0 by default.)
    majority_class = 1.0 if pos_count > neg_count else 0.0
    
    # Create baseline predictions: assign the majority class to every test edge.
    y_scores = np.full(y_true.shape, fill_value=majority_class, dtype=float)
    
    # Compute AUC and Average Precision.
    auc_score = roc_auc_score(y_true, y_scores)
    ap_score = average_precision_score(y_true, y_scores)
    
    print("Baseline Model: Predicting the Most Occurring Class")
    print(f"Test positive edge count: {pos_count}")
    print(f"Test negative edge count: {neg_count}")
    print(f"Predicted majority class: {majority_class}")
    print(f"AUC: {auc_score:.4f}")
    print(f"Average Precision (AP): {ap_score:.4f}")

if __name__ == "__main__":
    main()
