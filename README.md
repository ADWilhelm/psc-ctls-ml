This repository belongs to my master thesis. The goal is to analyze charge transport layers (CTL) of perovskite solar cells (PSC) using machine learning (ML).

Research Objective 1 compiles a dictionary of CTL names to SMILES codes. To reproduce the process, run the following files in order:

1. fetch_psc_data.ipynb: This will download the data from the NOMAD repository.
2. Describe_CTLs.ipynb: This is optional for inspecting the data.
3. Identify_CID_v2.ipynb: This is the core of Research Objective 1. It transforms CTL names to PubChem CIDs. To execute it, you will need API keys for groq and Elsevier, though.
4. Identification_results.ipynb: This checks how successful the identification was and creates a dataframe which is later used in Research Objective 2.
5. CID_to_SMILES.ipynb: This is optional and transforms the PubChem CIDs to SMILES codes.

Research Objective 2 compares three different machine learning approaches for predicting PCE using the CTL informaition gained in Research Objective 1. First, you should run data_prep_ML.ipynb, which performs some general data preparation for all ML approaches. Afterwards, you can run either of the three ML files: CrabNet.ipynb, XGBoost.ipynb, GNN_prediction.ipynb. The GNN models are collected within the folder "models". The XGBoost models are collected within the folder "models/trained_models".
