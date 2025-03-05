import numpy as np
import torch
import torch.nn.functional as F
import pandas as pd
from torch_geometric.nn import GCNConv, VGAE
import torch.nn as nn
from sklearn.metrics import precision_recall_curve, auc, roc_auc_score, average_precision_score



class GCNEncoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GCNEncoder, self).__init__()
        self.conv1 = GCNConv(in_channels, 2 * out_channels)
        self.conv2 = GCNConv(2 * out_channels, 2 * out_channels)  # Extra message passing
        self.conv_mu = GCNConv(2 * out_channels, out_channels)
        self.conv_logstd = GCNConv(2 * out_channels, out_channels)
        self.dropout = nn.Dropout(p=0.3)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = self.dropout(x)
        x = F.relu(self.conv2(x, edge_index))  # Additional message passing
        x = self.dropout(x)
        mu = self.conv_mu(x, edge_index)
        logstd = self.conv_logstd(x, edge_index)
        return mu, logstd  # Return both mean and log standard deviation


class VariationalGAE(VGAE):
    def __init__(self, encoder):
        super(VariationalGAE, self).__init__(encoder)

    def encode(self, x, edge_index):
        mu, logstd = self.encoder(x, edge_index)
        z = self.reparameterize(mu, logstd)

        # Fix: Store mu and logstd inside the model
        self.__mu__ = mu
        self.__logstd__ = logstd

        return z, mu, logstd

    def reparameterize(self, mu, logstd):
        """
        Implements the reparameterization trick: z = mu + std * epsilon
        where epsilon is a random sample from N(0, I).
        """
        std = torch.exp(logstd)  # Convert log-std to standard deviation
        eps = torch.randn_like(std)  # Sample epsilon from N(0, I)
        return mu + eps * std  # Apply reparameterization trick



def train(model, data, optimizer, version, patience=60, delta=0.01, checkpoint_path="best_model.pth"):
    best_auc = float('-inf')
    counter = 0
    num_epochs = 200
    
    auc_values = []
    ap_values = []
    precision_values = []
    recall_values = []
    
    latent_parameters = []  # Store mu and logstd for each epoch

    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        z, mu, logstd = model.encode(data.x, data.train_pos_edge_index)
        loss = model.recon_loss(z, data.train_pos_edge_index)
        loss += (1 / data.num_nodes) * model.kl_loss()
        loss.backward()
        optimizer.step()

        # Compute validation AUC, AP, Precision, Recall
        auc_score, ap_score, precision, recall = test(model, data)
        auc_values.append(auc_score)
        ap_values.append(ap_score)
        precision_values.append(precision)
        recall_values.append(recall)

        if ((epoch + 1) % 100 == 0):
            print(f"Epoch {epoch + 1}/{num_epochs} - Loss: {loss:.4f}, AUC: {auc_score:.4f}, AP: {ap_score:.4f}")

        # Save the latent parameters (mu, logstd) for each node in this epoch
        latent_parameters.append({
            "epoch": epoch + 1,
            "mu": mu.detach().cpu().numpy().tolist(),
            "logstd": logstd.detach().cpu().numpy().tolist()
        })

        # Early stopping logic
        if auc_score > best_auc + delta:
            best_auc = auc_score
            counter = 0
            torch.save(model.state_dict(), checkpoint_path)  # Save best model
        else:
            counter += 1
            if counter >= patience:
                print(f"Early stopping triggered at epoch {epoch + 1}. Best AUC: {best_auc:.4f}")
                break

    # Load the best model
    model.load_state_dict(torch.load(checkpoint_path))
    print(f"Best model restored with AUC: {best_auc:.4f}")

    # Save latent parameters to a CSV file

    save_csv_path=f"latent_parameters_v{version}.csv"
    save_latent_parameters(latent_parameters, save_csv_path, version)  # Pass version

    # Store metrics in a dictionary
    metrics_dict = {
        "AUC": auc_values,
        "AP": ap_values,
        "Precision": precision_values,
        "Recall": recall_values
    }
    
    return metrics_dict



def test(model, data):
    model.eval()
    with torch.no_grad():
        z, mu, logstd = model.encode(data.x, data.train_pos_edge_index)

        # Compute predicted scores
        pos_edge_scores = model.decoder(z, data.test_pos_edge_index).sigmoid().cpu().numpy()
        neg_edge_scores = model.decoder(z, data.test_neg_edge_index).sigmoid().cpu().numpy()

        # Ground truth labels (1 for positive edges, 0 for negative edges)
        y_true = np.concatenate([np.ones(pos_edge_scores.shape[0]), np.zeros(neg_edge_scores.shape[0])])
        y_scores = np.concatenate([pos_edge_scores, neg_edge_scores])

        # Compute AUC and AP
        auc_score = roc_auc_score(y_true, y_scores)
        ap_score = average_precision_score(y_true, y_scores)

        # Compute Precision-Recall curve
        precision, recall, _ = precision_recall_curve(y_true, y_scores)

    return auc_score, ap_score, precision, recall


def save_latent_parameters(latent_parameters, filename, version):
    """
    Saves the latent space parameters (mu, logstd) for each epoch into a CSV file.
    Prevents overwriting by appending data for different versions.
    """
    rows = []
    for entry in latent_parameters:
        epoch = entry["epoch"]
        for node_idx, (mu_val, logstd_val) in enumerate(zip(entry["mu"], entry["logstd"])):
            rows.append({
                "version": version,  # Add version column
                "epoch": epoch,
                "node": node_idx,
                "mu": mu_val,
                "logstd": logstd_val
            })

    df = pd.DataFrame(rows)

    # Append to CSV instead of overwriting
    try:
        existing_df = pd.read_csv(filename)
        df = pd.concat([existing_df, df], ignore_index=True)
    except FileNotFoundError:
        pass  # File doesn't exist yet, create a new one

    df.to_csv(filename, index=False)
    print(f"Saved latent parameters for version {version} to {filename}")
