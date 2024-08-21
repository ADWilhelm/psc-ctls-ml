This repository belongs to my master thesis written at Loughborough University. 
# Efficiency Prediction for Perovskite Solar Cells Using Automated Curation of Charge Transport Layer Material Properties

The goal was to analyze charge transport layers (CTL) of perovskite solar cells (PSC) using machine learning (ML).

For reproducing, in theory you would need to run all files in order. However, the resulting files are each already saved in their respective data folders ("data_RO1", "data_RO2", "data_RO3"), so you can start at any point and any file should run as it is.

## Research Objective 1
Research Objective 1 compiles a dictionary of CTL names to SMILES codes. To reproduce the process, run the following files in order:

1. fetch_PSC_data.ipynb: This will download the data from the NOMAD repository. It may take a while.
2. describe_CTLs.ipynb: This is optional for inspecting the data.
3. identify_CID_v2.ipynb: This is the core of Research Objective 1. It transforms CTL names to PubChem CIDs. To execute it, you will need an api_keys.py file containing API keys for groq and Elsevier, though, which is not uploaded here due to data protection reasons.
4. identification_results.ipynb: This checks how successful the identification was and creates a dataframe which is later used in Research Objective 2.
5. CID_to_SMILES.ipynb: This transforms the PubChem CIDs to SMILES codes.

## Research Objective 2
Research Objective 2 compares three different machine learning approaches for predicting PCE using the CTL information gained in Research Objective 1. The following notebooks are contained:

1. data_prep_ML.ipynb, which performs some general data preparation for all ML approaches
2. XGBoost.ipynb (The XGBoost models are collected within the folder "models/xgboost_models".)
3. CrabNet.ipynb: Run CrabNet predictions (The CrabNet models are collected within the folder "models/trained_models")
4. GNN_prediction.ipynb (The GNN models are collected within the folder "models".)
5. model_comparison_and_plots.ipynb: statistical evaluation of prediction errors and some nice visuals

## Research Objective 3
Research Objective 3 was the creation of an application combining the results of Objective 2 and 3. Its files consist of:

1. CTL_selection_helper_tool.ipynb: The helper tool inside a jupyter notebook. Example data for exploring the application is included below the application code.
2. helper_tool.py: The same helper tool as a python script for cleaner execution.
Again, an api_keys.py file will be necessary to use these. It needs to contain API_keys for Elsevier and Groq.

If you encounter problems, please let me know. I am happy to help. If you want to test the application but have trouble acquiring the necessary API keys, also let me know. I can easily help with that, I just cannot upload them anywhere, especially not on the public GitHub repository.
