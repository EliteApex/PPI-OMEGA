# PPI-OMEGA

This is the project repository for Data Science Capstone at UCSD in 2025 by Team A06-1.

Protein-Protein Interaction with Omics-Enhanced Graph Autoencoder, a.k.a. PPI-OMEGA, is a Variational Graph Autoencoder (VGAE)-based framework designed to improve Protein-Protein Interaction (PPI) predictions by integrating multi-omics data. Unlike traditional models that rely solely on static network topology, PPI-OMEGA incorporates RNA expression profiles and protein expression data to learn biologically meaningful representations.

---

## Installation and Execution Guide

### **Option 1: Running with Docker (Recommended)**
If you prefer a pre-configured environment, you can use Docker.

#### **1. Install Docker**
Ensure you have Docker installed. You can download it from [here](https://www.docker.com/get-started/).

#### **2. Pull the Docker Image**
You can pull the pre-built Docker image directly (if it's available on Docker Hub):
```bash
docker pull eliteapex/ppi-omega
```
Alternatively, you can build the image manually:
```bash
git clone https://github.com/EliteApex/PPI-OMEGA.git
cd PPI-OMEGA
docker build -t ppi-omega .
```

#### **3. Run the Docker Container**
Run the container interactively:
```bash
docker run -it --rm -v $(pwd):/app ppi-omega bash
```
This will mount your current directory (`PPI-OMEGA`) inside the container, so you can access scripts and data.

#### **4. Run the Model inside Docker**
Inside the container:
```bash
python src/run_models.py --version <version_num>
```
where `<version_num>` = 1, 2, or 3 depending on the input features you'd like to use.

#### **5. Using VS Code with Docker**
To use VS Code with the Docker container:
1. Install the [Remote - Containers](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers) extension.
2. Open VS Code and connect to the container:
   - Open Command Palette (`Ctrl+Shift+P`).
   - Select **Remote-Containers: Attach to Running Container**.
   - Choose `ppi-omega` from the list.
3. You can now use VS Code as if working in a local environment.

---

### **Option 2: Running Locally with Conda**
If you don't want to use Docker, you can manually set up the environment.

#### **1. Clone the Repository**
```bash
git clone https://github.com/EliteApex/PPI-OMEGA.git
cd PPI-OMEGA
```

#### **2. Set Up Conda Environment**
Ensure that Conda is installed. Then, create and activate the environment:
```bash
conda env create -f environment.yml
conda activate PPIOMEGA_env  
```

#### **3. Run the Model**
Once the environment is set up, you can run the model:
```bash
python src/run_models.py --version <version_num>
```

or within a Jupyter Notebook:
```python
%run src/run_models.py --version <version_num>
```
where `<version_num>` = 1, 2, or 3 depending on the input features you'd like.

---

## Project Structure

The repository is organized as follows:

```text
.
├── Data/                      # Dataset directory storing original and intermediate data files
│   ├── raw/                   # Raw data files
│   │   ├── normal_ihc_data.tsv
│   │   ├── protein_gene_conversion.csv
│   │   ├── rna_tissue_gtex.tsv
│   ├── adj_matrix_scaled.npz
│   ├── adj_matrix.npz
│   ├── filtered_PPI.csv
│   ├── PPI_protein_expression_full.csv
│   ├── PPI_Protein_only.csv
│   ├── PPI_RNA_only.csv
│   ├── PPI_RNA_Protein_combined.csv
│   ├── PPI_RNA_seq_full.csv
│   ├── protein_gene_conversion.csv
│   ├── protein_node_id_conversion.csv
│   └── protein_vis_samples.csv
├── notebooks/                 # Jupyter notebooks for analysis and visualization
│   ├── EDA.ipynb
├── plots/                     # Directory for storing plots and visualizations
├── scripts/                   # Additional scripts for data processing
│   ├── preprocessing.py
│   ├── vis_ppi_network_sample_data.py
├── src/                       # Source code for the project
│   ├── _pycache__/            # Cached Python files
│   ├── baseline_model.py
│   ├── best_hyperparameters.csv
│   ├── best_model.pth
│   ├── latent_parameters_v0.csv
│   ├── latent_parameters_v1.csv
│   ├── latent_parameters_v2.csv
│   ├── latent_parameters_v3.csv
│   ├── latent_variables_sampled.csv
│   ├── metrics_version_0.npz
│   ├── metrics_version_1.npz
│   ├── metrics_version_2.npz
│   ├── metrics_version_3.npz
│   ├── model.py
│   ├── pilot_with_features.ipynb
│   ├── run_models.py
│   ├── selected_nodes.txt
│   ├── vgae_cv.py
│   └── visualization.ipynb
├── .dockerignore              # Files and directories to ignore in Docker builds
├── .gitignore                 # Git ignore rules
├── Dockerfile                 # Docker container definition
├── environment.yml            # Conda environment file
└── README.md                  # Documentation

```

---

## Contributors
- **Team A06-1** - UCSD Data Science Capstone 2025
   - <a href="https://www.linkedin.com/in/siddharth-vyasabattu/">Siddharth Vyasabattu</a>
   - <a href = "https://www.linkedin.com/in/xiaoyu-gui-654020251/">Xiaoyu Gui</a>


---
