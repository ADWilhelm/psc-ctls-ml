This repository belongs to my master thesis. The goal is to analyze charge transport layers (CTL) of perovskite solar cells (PSC) using machine learning (ML).

# Step 1 
Fetch the PSC data from the NOMAD database, in which 43 119 PSCs are documented.
Files:
    
    - fetch_psc_data.ipynb

# Step 2 
Identify PubChem CIDs for the CTL materials. For materials with non-chemical names this is done using Llama3-70b-8192. With this pipeline, 825 of the 2559 materials can be identified which accounts for
80-90% of cells.
Files:

    - Identify_CID_v2.ipynb (find CIDs for the CTL materials. The core part of data curation.)
    - How_many_cells_identified.ipynb (for checking the results of the identification pipeline)
    - CID_to_SMILES.ipynb (transform CIDs to SMILES representation which is more suitable for ML.)

# Step 3 
Machine Learning analyses for PCE prediction.

Model_1_CrabNet_with_Label_Encoding.ipynb contains a PCE prediction using the absorber layer as formula and the Etl, HTL and device stack information as additionial variables for the model. The results are middling.

Model_2_CrabNet_CrabNet_with_device_stack_formulas.ipynb contains a PCE prediction for which the device stack materials were transformed into chemical formulas and then entirely used as formula for prediction.

Model_3_Graph_Neural_Network.ipynb contains a PCE prediction using a graph neural network, which interprets the molecules as nets of featurized nodes and edges.


