import pandas as pd

# assume the file 9606.protein.info.v12.0.txt from STRING database is stored in the same directory
protein_info = pd.read_csv('9606.protein.info.v12.0.txt', sep='\t')

antigen_keywords = "antibod|antigen|immunoglobulin|b-cell receptor"
antibody_df = protein_info[protein_info['annotation'].str.lower().str.contains(antigen_keywords, regex=True)]

transporter_keywords = "transporter|pump|channel|carrier protein|solute carrier|abc transporter|ion channel|membrane transport"
transporter_df = protein_info[protein_info['annotation'].str.lower().str.contains(transporter_keywords, regex=True)]

enzyme_keywords = "kinase|phosphatase|hydrolase|synthase|dehydrogenase|catalytic activity|protease|lyase|oxidoreductase"
enzyme_df = protein_info[protein_info['annotation'].str.lower().str.contains(enzyme_keywords, regex=True)]

receptor_keywords = "receptor|g-protein coupled receptor|gpcr|tyrosine kinase receptor|cytokine receptor|toll-like receptor|nuclear receptor|signaling receptor"
receptor_df = protein_info[protein_info['annotation'].str.lower().str.contains(receptor_keywords, regex=True)]

print("Antibodies:", antibody_df.shape[0])
print("Transporters:", transporter_df.shape[0])
print("Enzymes:", enzyme_df.shape[0])
print("Receptors:", receptor_df.shape[0])

# Convert each category's protein IDs to a set
antibody_set = set(antibody_df["#string_protein_id"])
transporter_set = set(transporter_df["#string_protein_id"])
enzyme_set = set(enzyme_df["#string_protein_id"])
receptor_set = set(receptor_df["#string_protein_id"])

# Get unique proteins for each category (excluding overlaps)
antibody_unique = antibody_set - (transporter_set | enzyme_set | receptor_set)
transporter_unique = transporter_set - (antibody_set | enzyme_set | receptor_set)
enzyme_unique = enzyme_set - (antibody_set | transporter_set | receptor_set)
receptor_unique = receptor_set - (antibody_set | transporter_set | enzyme_set)

# Convert sets to lists for sampling
antibody_sample = pd.DataFrame({"#string_protein_id": list(antibody_unique)[:250], "group": "Antibody"})
transporter_sample = pd.DataFrame({"#string_protein_id": list(transporter_unique)[:250], "group": "Transporter"})
enzyme_sample = pd.DataFrame({"#string_protein_id": list(enzyme_unique)[:250], "group": "Enzyme"})
receptor_sample = pd.DataFrame({"#string_protein_id": list(receptor_unique)[:250], "group": "Receptor"})

# Concatenate into one DataFrame
sampled_df = pd.concat([antibody_sample, transporter_sample, enzyme_sample, receptor_sample], ignore_index=True)
sampled_df.columns = ['protein_id', 'group']
sampled_df.to_csv('../Data/protein_vis_samples.csv', index=False)