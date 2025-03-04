import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from scipy.sparse import save_npz, coo_matrix

from pyensembl import EnsemblRelease # Python interface to Ensembl reference genome metadata
match_data = EnsemblRelease(104)
match_data.download()
match_data.index()

## Helper functions
def get_pc(mat, num=10):
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(mat)
    pca = PCA(n_components=num)  
    principal_components = pca.fit_transform(df_scaled)
    return principal_components, pca

def find_common_prefix(strings):
    if not strings:
        return ""
    prefix = strings[0]
    for string in strings[1:]:
        while not string.startswith(prefix):
            prefix = prefix[:-1]
            if not prefix:
                return ""
    return prefix


# Protein-Protein Interaction Dataset
ppi_data = pd.read_csv('../Data/9606.protein.links.v12.0.txt', sep='\t', names=['link_info'])
ppi_data = ppi_data.loc[1:]
ppi_data[['protein1', 'protein2', 'combined_score']] = ppi_data['link_info'].str.split(expand=True)
ppi_data.drop(columns=['link_info'], inplace=True)
ppi_data['combined_score'] = ppi_data['combined_score'].astype(int)

## Threshold and filter for top 5% scored interactions
threshold = np.percentile(ppi_data['combined_score'], 95)
filtered_ppi_data = ppi_data[ppi_data['combined_score'] >= threshold]

filtered_ppi_data.to_csv('../Data/filtered_PPI.csv',index=False)


# RNA-seq Dataset
## The data is sourced and unzipped from https://www.proteinatlas.org/download/tsv/rna_tissue_gtex.tsv.zip
rna_df = pd.read_csv('../Data/rna_tissue_gtex.tsv', sep='\t')
rna_df = rna_df[['Gene','Tissue','TPM']] # keep necessary columns

gene_tissue_pivot = rna_df.pivot_table(
    index='Gene',       
    columns='Tissue',   
    values='TPM',       
    aggfunc='mean'      
) # pivot table of Gene x Tissue
gene_tissue_pivot.dropna(inplace=True)
gene_tissue_pivot.reset_index(inplace=True)

## Match Gene IDs in RNA dataset to Protein IDs
all_proteins = set(ppi_data['protein1']).union(set(ppi_data['protein2']))
all_proteins = [i.replace('9606.','') for i in all_proteins]

protein_to_gene = {}
for protein in all_proteins:
    try:
        gene_id = match_data.gene_id_of_protein_id(protein)
        protein_to_gene[protein] = gene_id
    except ValueError:
        continue

protein_gene_conversion = pd.DataFrame(protein_to_gene.keys(),protein_to_gene.values()).reset_index()
protein_gene_conversion.columns = ['Gene', 'Protein']
protein_gene_conversion.to_csv('../Data/protein_gene_conversion.csv', index=False)

match_feature_mat = gene_tissue_pivot.merge(protein_gene_conversion, on='Gene', how='right')
print(f"Number of unmatched proteins: {match_feature_mat.amygdala.isna().sum()}")
match_feature_mat = match_feature_mat.dropna().drop(columns = ['Gene'])
match_feature_mat = match_feature_mat[['Protein']+list(match_feature_mat.columns[:-1])]
print(f"Feature matrix created for {match_feature_mat.shape[0]} proteins.")

## Create Feature Matrix
features_only = match_feature_mat.iloc[:,1:].values
principal_components_rna, pca_rna = get_pc(features_only)
print("Explained variance ratio:", pca_rna.explained_variance_ratio_)
print("Cumulative explained variance:", pca_rna.explained_variance_ratio_.cumsum())
protein_keep = match_feature_mat['Protein']
rna_pca_df = pd.DataFrame(principal_components_rna)
rna_pca_df.index = protein_keep
match_feature_mat.to_csv("../Data/PPI_RNA_seq_full.csv", index=False)
print("Percentage of proteins after preprocessing with proper RNA feature: " + str(match_feature_mat.shape[0]/(ppi_data['protein1'].nunique())))


# IHC Protein Expression Dataset
## The data is sourced and unzipped from https://www.proteinatlas.org/download/tsv/normal_ihc_data.tsv.zip 
protein_df = pd.read_csv('../Data/normal_ihc_data.tsv', sep='\t')
protein_df.dropna(inplace=True)

## Encode `Level` column
keep_levels_dict = {'Not detected':0, 'Low':1, 'Medium':2, 'High':3}
print("Number of rows dropped: " + str(protein_df[~protein_df['Level'].isin(keep_levels_dict.keys())].shape[0]))
protein_df = protein_df[protein_df['Level'].isin(keep_levels_dict.keys())]
protein_df['Level_OHE'] = protein_df['Level'].map(keep_levels_dict)
protein_df['Tissue_CellType'] = protein_df['IHC tissue name'] + ' / ' + protein_df['Cell type']

combination_info = protein_df.groupby(['IHC tissue name','Cell type'])['Gene'].count().reset_index()
combined_keep = combination_info[combination_info['Gene']>13000]
combinations_keep = combined_keep['IHC tissue name'] + ' / ' + combined_keep['Cell type']
protein_df_keep_comb = protein_df[protein_df['Tissue_CellType'].isin(combinations_keep.values)]

ohe_df = protein_df_keep_comb.pivot_table(
    index='Gene',
    columns='Tissue_CellType',
    values='Level_OHE',
    aggfunc='max'  
) # pivot table of Protein x Tissue/Cell type
ohe_df.reset_index(inplace=True)
ohe_df_no_missing = ohe_df.dropna()

match_feature_mat_protein = ohe_df_no_missing.merge(protein_gene_conversion, on='Gene', how='right')
print(f"Number of unmatched proteins: {ohe_df_no_missing['Adipose tissue / adipocytes'].isna().sum()}")
match_feature_mat_protein = match_feature_mat_protein.dropna().drop(columns = ['Gene'])
match_feature_mat_protein = match_feature_mat_protein[['Protein']+list(match_feature_mat_protein.columns[:-1])]
print(f"Feature matrix created for {match_feature_mat_protein.shape[0]} proteins.")

## Create Feature Matrix
features_only_protein = match_feature_mat_protein.iloc[:,1:].values
principal_components_protein, pca_protein = get_pc(features_only_protein,10)
print("Explained variance ratio:", pca_protein.explained_variance_ratio_)
print("Cumulative explained variance:", pca_protein.explained_variance_ratio_.cumsum())
protein_keep_feature = match_feature_mat_protein['Protein']
pe_pca_df = pd.DataFrame(principal_components_protein)
pe_pca_df.index = protein_keep_feature
match_feature_mat_protein.to_csv("../Data/PPI_protein_expression_full.csv", index=False)


# Prepare data for model
## Get rid of common prefix
protein1_prefix = find_common_prefix(filtered_ppi_data['protein1'].tolist())
protein2_prefix = find_common_prefix(filtered_ppi_data['protein2'].tolist())
print("Common prefix in 'protein1':", protein1_prefix)
print("Common prefix in 'protein2':", protein2_prefix)
rna_seq_prefix = find_common_prefix(rna_pca_df.index.tolist())
print("Common prefix in 'rna seq proteins':", rna_seq_prefix)
pe_prefix = find_common_prefix(pe_pca_df.index.tolist())
print("Common prefix in 'protein expression proteins':", pe_prefix)
assert protein1_prefix == protein2_prefix
assert rna_seq_prefix == pe_prefix

filtered_ppi_data['protein1'] = filtered_ppi_data['protein1'].str.replace(protein1_prefix, '', regex=False)
filtered_ppi_data['protein2'] = filtered_ppi_data['protein2'].str.replace(protein1_prefix, '', regex=False)
rna_pca_df.index = rna_pca_df.index.str.replace(rna_seq_prefix, '', regex=False)
pe_pca_df.index = pe_pca_df.index.str.replace(pe_prefix, '', regex=False)

## Prepare for combined feature matrix
feature_df = rna_pca_df.merge(pe_pca_df, on = 'Protein')
valid_nodes = set(feature_df.index)
ppi_df_filtered = filtered_ppi_data[filtered_ppi_data['protein1'].isin(valid_nodes) & filtered_ppi_data['protein2'].isin(valid_nodes)]
filtered_nodes = pd.concat([ppi_df_filtered['protein1'], ppi_df_filtered['protein2']]).unique()
print(f"Number of proteins with valid features: {filtered_nodes.shape[0]}")

node_to_idx = {node: i for i, node in enumerate(filtered_nodes)}
ppi_df_filtered['protein1_idx'] = ppi_df_filtered['protein1'].map(node_to_idx)
ppi_df_filtered['protein2_idx'] = ppi_df_filtered['protein2'].map(node_to_idx)
adj_matrix = coo_matrix(
    (ppi_df_filtered['combined_score'], (ppi_df_filtered['protein1_idx'], ppi_df_filtered['protein2_idx'])),
    shape=(len(filtered_nodes), len(filtered_nodes))
)

### Minmax scale the protein interaction scores
scaler = MinMaxScaler()
ppi_df_filtered['score_scaled'] = scaler.fit_transform(ppi_df_filtered[['combined_score']])
adj_matrix_scaled = coo_matrix(
    (ppi_df_filtered['score_scaled'], (ppi_df_filtered['protein1_idx'], ppi_df_filtered['protein2_idx'])),
    shape=(len(filtered_nodes), len(filtered_nodes))
)

feature_df['node_idx'] = feature_df.index.map(node_to_idx)
feature_df_filtered = feature_df.dropna(subset=['node_idx']).sort_values(by='node_idx')
feature_matrix = feature_df_filtered.iloc[:, :-1].values  # exclude index 'Protein' and 'node_idx'
assert adj_matrix.shape[0] == feature_matrix.shape[0]

feature_mat_processed = pd.DataFrame(feature_matrix)
feature_mat_processed.to_csv("../Data/PPI_RNA_Protein_combined.csv", index=False)
rna_only = feature_mat_processed.iloc[:,:10]
protein_only = feature_mat_processed.iloc[:,10:]
rna_only.to_csv("../Data/PPI_RNA_only.csv", index=False)
protein_only.to_csv("../Data/PPI_Protein_only.csv", index=False)
save_npz("../Data/adj_matrix.npz", adj_matrix)
save_npz("../Data/adj_matrix_scaled.npz", adj_matrix_scaled)

# To load the data:
# adj_mat_scaled = np.load('../Data/adj_matrix_scaled.npz')
# feature_mat_combined = pd.read_csv('../Data/PPI_RNA_Protein_combined.csv')