import httpx
import api_keys # file containing a groq and elsevier api-key
import shutil
import requests
import pickle
import os
import json
import torch
import ast
import re

import ipywidgets as widgets
import numpy as np
import pandas as pd
import torch.nn.functional as F

from torch_geometric.nn import MessagePassing, global_mean_pool, MLP, global_add_pool
from torch_geometric.nn import GCNConv
from rdkit.Chem import AllChem, rdFingerprintGenerator
from rdkit import Chem, DataStructs
from xgboost import XGBRegressor
from groq import Groq
from IPython.display import clear_output
from ipywidgets import Button, Layout

shutil.copyfile("../ResObj_2/models/xgboost_models/xgboost_full_model.json", "data_RO3/xgboost_full_model.json")
shutil.copyfile("../ResObj_2/data_RO2/SMILES_dictionary.pkl", "data_RO3/SMILES_dictionary.pkl")

def find_CID_in_text(industry_name, 
                     CTL_DOIs, 
                     paper_index = -1,
                     depth = 1):
    '''
    This function tries to find the chemical name in the paper text and to
    retrieve the CID from PubChem.
    Arguments: 
        industry_name (str) - the name of the material
        CTL_DOIs (list) - list of DOIs of papers with that CTL
        paper_index (int) - index of the paper in the list
    Value: 
        CID (int) - the CID of the material
    Dependencies: 
        paper_from_publisher, llm_retrieve_name, search_pubchem_by_name
    '''
    # necessary not to break execution due to too many recursions. 
    # Adjust to suit your machine's recursion limit.
    if depth > 2900:
        print(f'debug: Recursion limit reached at recusion = {depth}.')
        raise Exception('Recursion limit reached')

    print(f"debug: Recursion depth: {depth}")
    print(f'debug: length of CTL_DOIs: {len(CTL_DOIs)}. Paper Index: {paper_index+1}')
    
    # stop if all papers for the material have been searched
    if paper_index >= (len(CTL_DOIs) - 1):
        raise Exception('All papers have been searched.')

    paper_index = paper_index + 1
    DOI = CTL_DOIs[paper_index]
    
    # Recursion section
    try:
        paper_text = paper_from_publisher(DOI)
    except:
        print('debug: paper_from_publisher failed. Recurring...')
        CID = find_CID_in_text(industry_name, CTL_DOIs, paper_index, depth+1)
        if CID is not None:
            return CID
    try:
        print('debug: trying llm_retrieve_name')
        chem_name = llm_retrieve_name(paper_text, industry_name)
    except:
        print('debug: llm_retrieve_name failed. Recurring...')
        CID = find_CID_in_text(industry_name, CTL_DOIs, paper_index, depth+1)
        if CID is not None:
            return CID
    try:
        print('debug: trying search_pubchem_by_name(chem_name)')
        CID = search_pubchem_by_name(chem_name, when_called="LLMresult")
        print('debug: search_pubchem_by_name with name found by LLM SUCCESSFUL!')
        return CID
    except:
        print('debug: trying search_pubchem_by_name(chem_name) failed. Recurring...')
        CID = find_CID_in_text(industry_name, CTL_DOIs, paper_index, depth+1)
        if CID is not None:
            return CID



def paper_from_publisher(DOI):
    '''
    This function tries publisher APIs to retrieve paper texts from ScienceDirect.
    Argument: DOI (str) - the DOI of the paper
    Value: paper_text (str) - the plain text of the paper
    '''

    def scidir_retrieve_paper(DOI, apikey):
        '''
        This function is used within paper_from_publisher to retrieve the paper
        text from ScienceDirect.
        '''
        apikey=apikey
        headers={
            "X-ELS-APIKey":apikey,
            "Accept":'application/json'
            }
        client = httpx.Client(headers=headers)
        query="&view=FULL"
        url=f"https://api.elsevier.com/content/article/doi/" + DOI
        r=client.get(url)
        print(f'debug: paper retrieval executed. This is the result: {r}')
        if r.status_code != 200:
            raise Exception(f"Error: The paper could not be found in ScienceDirect. Status code: {r.status_code}")
        return r

    # Get document
    try:
        scidir_response = scidir_retrieve_paper(DOI, api_keys.api_key_elsevier)

        json_acceptable_string = scidir_response.text
        d = json.loads(json_acceptable_string)
        return d['full-text-retrieval-response']['coredata']['dc:description']
    except:
        print("debug: Paper not found in ScienceDirect.")
        raise Exception("Error: Paper not found in ScienceDirect.")



def llm_retrieve_name(paper_text,
                      industry_name, 
                      api_key=api_keys.api_key_groq):
    '''
    This function retrieves the chemical name of the compound from the paper text.
    Arguments: 
        paper_text (str) - the text of the paper
        industry_name (str) - the name of the material
        api_key (str) - an api_key for groq
    Value: 
        chem_name (str) - the chemical name of the material
    '''
    groq = Groq(api_key=api_key)
    chat_completion = groq.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": "You are a solar cell scientist proficient in reading papers. You output only the chemical name of the compound asked for, nothing else.",
            },
            {
                "role": "user",
                "content": f"What is the chemical name pertaining to this abbreviation: {industry_name}? You can find it in this text: {paper_text}.",
            }
        ],
        model="llama3-70b-8192",
    )
    return chat_completion.choices[0].message.content


def search_pubchem_by_name(industry_name, when_called = "initial"):
    '''
    This searches for a CTL material's CID in PubChem.
    Arguments:  industry_name (str) - the name of the material
                when_called (str) - "initial" or other; used to determine which 
                                    counter to increment
    Value: CID (int) - the CID of the material
    '''
    
    url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{industry_name}/cids/JSON"
    
    response = requests.get(url)
    
    if response.status_code == 200:
        data = response.json()
        data = data['IdentifierList']['CID'][0]
        if when_called == "initial":
            print("SUCCESS: Initial compound search")
        else:
            print("SUCCESS: LLM result compound search")
        return data
    else:
        # if search in PubChem compounds fails, search in substances
        url_subs = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/substance/name/{industry_name}/cids/JSON"

        response = requests.get(url_subs)

        if response.status_code == 200:
            data = response.json()
            data = data['InformationList']['Information'][0]['CID'][0]
            print("Substance search SUCCESSFUL")
            if when_called == "initial":
                print("SUCCESS: Initial substance search")
            else:
                print("SUCCESS: LLM result substance search")
            return data
        else:
            # if the search failed again, report failure
            print('debug: substance search unsuccessful')
            raise Exception(f"Error: Could not retrieve SID using the industry name alone. Status code: {response.status_code}")
            
def CID_to_SMILES(CID):
    '''
    This transforms a CID into a SMILES using PubChem.
    Argument: CID (int) - the CID of the material
    data - the SMILES of the material
    '''
    url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{CID}/property/CanonicalSMILES/JSON"

    response = requests.get(url)

    if response.status_code == 200:
        data = response.json()
        data = data['PropertyTable']['Properties'][0]['CanonicalSMILES']
        return data
    else:
        print('debug: SMILES could not be retrieved')
        raise Exception(f"Error: Could not retrieve SMILES from this CID. Status code: {response.status_code}")
        return None
    
    

def LLM_transformation(industry_name, doi_list):
    
    doi_list = [doi.strip() for doi in doi_list.split(',')]
    
    industry_name = "['" + industry_name + "']"
    industry_name = ast.literal_eval(industry_name)[0]

    # check if name already in dictionary
    try:
        with open('data_RO3/SMILES_dictionary.pkl', 'rb') as f:
            SMILES_dict = pickle.load(f)
    except:
        print("Could not access SMILES_dictionary. Make sure it is in data_RO2/SMILES_dictionary.pkl.")

    # if already in dictionary, do no more
    if industry_name in SMILES_dict.keys() and SMILES_dict[industry_name] is not None:
        return SMILES_dict[industry_name]

    else:
        print("This material is not yet in our dictionary. Attempting extraction with the DOIs you provided.")
        CID = None
        # procedure that tries to find the CID for the industry_name
        try:
            CID = search_pubchem_by_name(industry_name, when_called="initial")
            print('Initial search_pubchem_by_name was SUCCESSFUL!')
            try:
                SMILES_dict[industry_name] = CID_to_SMILES(CID)
            except:
                print("Could not perform CID_to_SMILES")
            print('New material added to dictionary')
            return SMILES_dict[industry_name]
        except:
            print('Initial search_pubchem_by_name was unsuccessful. Trying paper reading...')

            # build list of DOIs that mention the material
            if len(doi_list) == 0:
                print('Initial search was unsuccessful and without dois, there is nothing I can do.')
                return

            else:
                print('Searching through papers...')
            # go through the list of papers and try llm extraction
            try:
                CID = find_CID_in_text(industry_name, CTL_DOIs=doi_list, depth=1)
                print(f"LLM found a result: {CID}")
            except Exception as e:
                if str(e) == 'All papers have been searched.':
                    print('All papers you provided have been searched, without success.')
                    raise Exception("All papers have been searched without success.")
                elif str(e) == 'Recursion limit reached':
                    raise Exception('The recursion limit (2990) was reached before extraction was completed.')
                else:
                    raise Exception("An unknown exception occured during paper reading.")
            try:
                SMILES_dict[industry_name] = CID_to_SMILES(CID)
            except:
                print("Could not perform CID_to_SMILES after paper reading.")
            return SMILES_dict[industry_name]

        
# XGBoost prediction

def xgb_make_predictions(bandgap, absorber, etl, htl, rank_what):

    if rank_what == "etl":
        htl = "['" + htl + "']"
        etl = "[" + etl + "]"
    else:
        etl = "['" + etl + "']"
        htl = "[" + htl + "]"
    
    try: 
        bandgap = float(bandgap)
    except:
        print("There is something wrong with the bandgap. Perhaps you used a comma as decimal separator?")
        return
    try:
        etl = ast.literal_eval(etl)
    except:
        print("There is something wrong with your ETL(s). Please correct and try again.")
        return
    try:
        htl = ast.literal_eval(htl)
    except:
        print("There is something wrong with your HTL(s). Please correct and try again.")
        return
    
    df = pd.DataFrame()
    
    if rank_what == "htl":
        df['htl'] = htl
        etl_list = []
        etl_list.append(etl[0])
        etl_list_list = []
        for index, row in df.iterrows():
            etl_list_list.append(etl_list)
            df['absorber'] = absorber
            df['bandgap'] = bandgap
        df['etl'] = etl_list_list
    elif rank_what == "etl":
        df['etl'] = etl
        htl_list = []
        htl_list.append(htl[0])
        htl_list_list = []
        for index, row in df.iterrows():
            htl_list_list.append(htl_list)
            df['absorber'] = absorber
            df['bandgap'] = bandgap
        df['htl'] = htl_list_list
    else:
        raise("Something went really wrong. Try again.")
    
    # prepare absorber
    def parse_formula(formula):
        matches = re.findall(r'([A-Z][a-z]?)(\d*)', formula)
        element_counts = {}
        for (element, count) in matches:
            if element in element_counts:
                element_counts[element] += int(count) if count else 1
            else:
                element_counts[element] = int(count) if count else 1
        return element_counts
    
    # Unique elements across all formulas
    ELEMENTS = ['H', 'C', 'N', 'O', 'F', 'P', 'S', 'Cl', 'Br', 'I', 'B', 'Li',
            'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ti', 'V', 'Cr', 'Mn', 'Fe',
            'Co', 'Ni', 'Cu', 'Zn', 'Ga', 'Ge', 'As', 'Se', 'Rb', 'Sr', 'Y',
            'Zr', 'Nb', 'Mo', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn', 'Sb',
            'Te', 'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd', 'Sm', 'Eu', 'Gd', 'Tb',
            'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu', 'Hf', 'Ta', 'W', 'Re', 'Os', 
            'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi', 'Th', 'U']

    # Create columns for each element count
    try:
        for element in ELEMENTS:
            df[element] = df['absorber'].apply(lambda x: parse_formula(x).get(element, 0))
    except:
        print("There is something wrong with the absorber input. Remember that the input should be a chemical "+
              "formula (e.g., CH6I3NPb), NOT the commonly used name (e.g., MAPbI3).")
    
    # prepare ctl
    df['etl_SMILES'] = df['etl']
    df['htl_SMILES'] = df['htl']
    etl_combined_SMILES = []
    htl_combined_SMILES = []
    
    #df['etl_SMILES'] = df['etl_SMILES'].apply(ast.literal_eval)
    #df['htl_SMILES'] = df['htl_SMILES'].apply(ast.literal_eval)
    
    for index, row in df.iterrows():
        etl_combination = ".".join(row['etl_SMILES'])
        htl_combination = ".".join(row['htl_SMILES'])
        etl_combined_SMILES.append(etl_combination)
        htl_combined_SMILES.append(htl_combination)
    df['etl_combined_SMILES'] = etl_combined_SMILES
    df['htl_combined_SMILES'] = htl_combined_SMILES
    
    df['etl_combined_SMILES'] = df['etl_combined_SMILES'].str.replace('no_ctl', '')
    df['htl_combined_SMILES'] = df['htl_combined_SMILES'].str.replace('no_ctl', '')
    
    
    
    fpgen = rdFingerprintGenerator.GetMorganGenerator(radius=2,fpSize=1024)    
        
    def smiles_to_fingerprint(smiles):
        molecule = Chem.MolFromSmiles(smiles)
        if molecule is None:
            return np.zeros(fpgen.GetNumBits(),)
        fp = fpgen.GetFingerprint(molecule)
        # Convert to a bit vector
        bit_vector = np.zeros((1,), dtype=int)
        DataStructs.ConvertToNumpyArray(fp, bit_vector)
        return bit_vector
    
    # Create columns for Morgan Fingerprints
    etl_fingerprints = df['etl_combined_SMILES'].apply(smiles_to_fingerprint)
    htl_fingerprints = df['htl_combined_SMILES'].apply(smiles_to_fingerprint)
    
    # Convert fingerprints to DataFrame
    etl_fingerprint_df = pd.DataFrame(etl_fingerprints.tolist(),
                                      columns=[f'ETL_FP_{i}' for i in range(1024)])
    htl_fingerprint_df = pd.DataFrame(htl_fingerprints.tolist(),
                                      columns=[f'HTL_FP_{i}' for i in range(1024)])
    
    # Combine all features into a single DataFrame
    features_df = pd.DataFrame
    features_df = pd.concat([df.drop(columns=['absorber',
                                          'etl', 
                                          'htl', 
                                          'etl_SMILES',
                                          'htl_SMILES',
                                          'etl_combined_SMILES',
                                          'htl_combined_SMILES']), 
                             etl_fingerprint_df, 
                             htl_fingerprint_df], 
                            axis=1)
    
    features_df = features_df.dropna()  
    
    # load model
    best_params = {'colsample_bytree': 0.8905864076209402, 
                   'gamma': 4.268246481233799, 
                   'learning_rate': 0.021626139436115823, 
                   'max_depth': 13, 
                   'n_estimators': 650, 
                   'subsample': 0.8026826799574963}
    optim_model = XGBRegressor(**best_params)
    optim_model.load_model('data_RO3/xgboost_full_model.json')
    predictions = optim_model.predict(features_df)
    df['predicted_PCE'] = predictions
    df_sorted = df.sort_values('predicted_PCE', ascending=False)
    
    ranking = pd.DataFrame()
    
    if rank_what == "htl":
        ranking['htl'] = df_sorted['htl']
    elif rank_what == "etl":
        ranking['etl'] = df_sorted['etl']
    
    ranking['Predicted PCE'] = df_sorted['predicted_PCE']
    display(ranking)
    
    
# Button handlers

def handler_use_as_ETL(b):
    with output:
        clear_output()
        try:
            xgb_make_predictions(absorber=abs_input.value,
                         bandgap=bg_input.value,
                         htl=ctl_input.value,
                         etl=list_input.value,
                         rank_what="etl")
        except:
            print("Prediction unsuccessful. Please make sure your input format is correct.")
        
def handler_use_as_HTL(b):
    with output:
        clear_output()
        try:
            xgb_make_predictions(absorber=abs_input.value,
                         bandgap=bg_input.value,
                         etl=ctl_input.value,
                         htl=list_input.value,
                         rank_what="htl")
        except:
            print("Prediction unsuccessful. Please make sure your input format is correct.")
            
            
def handler_identify_LLM(b):
    with output2:
        clear_output()
        try: 
            print(LLM_transformation(industry_name=LLM_material_input.value,
                                 doi_list=LLM_doi_input.value))
        except:
            print("LLM transformation unsuccessful.")


# input widgets-----------------------------------------------------------------

abs_input = widgets.Textarea(placeholder='e.g. CH6I3NPb...',
                            layout = Layout(width='auto'))
label_abs_input = widgets.HTML('<b>Absorber layer composition:</b><br> '+
                               'This needs to be a chemical formula, consisting of '+
                               'elements and their respective amount in the material')

bg_input = widgets.Text(placeholder='e.g. 1.56...',
                       layout = Layout(width='150px'))
label_bg_input = widgets.HTML(value='<b>Absorber band gap:</b>')

ctl_input = widgets.Textarea(placeholder="e.g. O=[Ti]=O...",
                            layout = Layout(width='auto'))
label_ctl_input = widgets.HTML(value='<b>Fixed CTL material (in SMILES format):</b><br> '+
                              'If you want to enter a multiple-material stack, separate '+
                              'the SMILES codes by a period, e.g. O=[Ti]=O.O=[Zn]')

list_input = widgets.Textarea(placeholder="e.g. \n['O=[Ti]=O'],\n['O=[Cr]O[Cr]=O', "+
                              "'O=[Zn]'],\n...",
                             layout = Layout(height='100px',
                                             width='auto',
                                               min_height='40px',
                                               overflow_y='auto'))
label_list_input = widgets.HTML('<b>Your material suggestions (in SMILES format):</b>')
label_list_expla = widgets.HTML("Each material stack needs to be enclosed in [' ']</b>")
label_list_expla2 = widgets.HTML('In a multiple-material stack, please separate individual '+
                                 'materials with a comma.')

#### LLM inputs
LLM_material_input = widgets.Textarea(placeholder="e.g. TiO2-c...",
                                 layout = Layout(width='auto'))
LLM_label_material_input = widgets.HTML(value='<b>Material for SMILES transformation:</b>')

LLM_doi_label = widgets.HTML(value='If we do not have a material in our dictionary, you can '+
                             'help us rectify that by providing DOIs of papers using that '+
                             ' material. In some cases, we are able to '+
                             'find the SMILES code using an LLM search. '+
                             'Please enter a list of DOIs (separated by commas).')
LLM_doi_input = widgets.Textarea(placeholder="e.g. 10.1038/s41560-021-00941-3, 10.1016/b9...",
                                 layout = Layout(width='auto'))

# Output widget----------------------------------------------------------------
output = widgets.Output()
output2 = widgets.Output()

# Button widgets ---------------------------------------------------------------
label_buttons = widgets.HTML('<style="color:#008b8b;><b>Predict PCE...</b>')

btn_use_as_ETL = widgets.Button(
    description='... using your materials as ETL',
    tooltip='Show the predicted best options for the HTL using the material as ETL',
    layout=widgets.Layout(width='auto', height='auto'),
    style=dict(font_style='italic',
              font_weight='bold',
              font_variant='small-caps',
              text_color='#008b8b'))

btn_use_as_HTL = widgets.Button(
    description='... using your materials as HTL',
    tooltip='Show the predicted best options for the ETL using the material as HTL',
    layout=widgets.Layout(width='auto', height='auto'),
    style=dict(font_style='italic',
              font_weight='bold',
              font_variant='small-caps',
              text_color='#008b8b'))

LLM_btn_identify = widgets.Button(
    description='Search SMILES',
    tooltip='Searches for a SMILES for your material',
    layout=widgets.Layout(width='auto', height='auto'),
    style=dict(font_style='italic',
              font_weight='bold',
              font_variant='small-caps',
              text_color='#008b8b'))

label_toggle_LLM = widgets.HTML(value='<b>SMILES or Common Names:</b>')


# Links ------------------------------------------------------------------------
btn_use_as_ETL.on_click(handler_use_as_ETL)
btn_use_as_HTL.on_click(handler_use_as_HTL)

LLM_btn_identify.on_click(handler_identify_LLM)


# Display ---------------------------------------------------------------------
# Aesthetic elements
line1 = widgets.HTML('<hr style="height:2px;background-color:#008b8b;">')
title = widgets.HTML('<h1 style="color:#008b8b;"><b>CTL Finder</b></h1>')
title2 = widgets.HTML('<h1 style="color:#008b8b;"><b>SMILES Translator</b></h1>')

# manual
manual = widgets.HTML('This program can help exlore CTL materials for perovskite solar cells.  '+
                      '<b>Assuming that you already know the absorber layer as well as one charge transport layer (CTL)  '+
                      'you wish to use (can be ETL or HTL), it ranks your materials suggestions for the respective other CTL by predicting PCE.</b>. '+
                      'To use it, follow these steps: <br>'+
                      '1.) Enter the absorber layer composition and bandgap <br>'+
                      '2.) Enter the fixed CTL, that is the one you already know you want to use (can be either HTL or ETL) <br>'+
                      '3.) Enter a list of the CTL for which you want to compare materials <br>'+
                      '4.) Start the prediction by clicking the respective button that will use "your materials" as either HTL or ETL. '+
                      'The fixed CTL will be interpreted as the respective other material.<br>'+
                      'Note that the CTLs will need to be input as SMILES codes. You can use the '+
                      'SMILES Translator to identify these or find them for example in PubChem.')

# CTL finder functional elements
ctl_finder_container = widgets.VBox([title, 
                                     manual,
                                     label_abs_input,
                                     abs_input,
                                     label_bg_input, 
                                     bg_input])

btn_container = widgets.HBox([
    label_buttons,
    btn_use_as_ETL,
    btn_use_as_HTL])

SMILES_container = widgets.VBox([label_ctl_input, 
                                 ctl_input,
                                 label_list_input,
                                 label_list_expla,
                                 label_list_expla2,
                                 list_input,
                                 btn_container,
                                 output], layout=Layout(width="auto"))

LLM_container = widgets.VBox([line1,
                              title2,
                              LLM_label_material_input, 
                              LLM_material_input,
                              LLM_doi_label,
                              LLM_doi_input,
                              LLM_btn_identify,
                              output2])

all_container = widgets.VBox([ctl_finder_container,
                              SMILES_container,
                              LLM_container],
                            layout=Layout(border='solid 5px #008b8b', padding='7px'))
# SMILES finder functional elements

display(all_container)