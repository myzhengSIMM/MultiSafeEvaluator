from model import *
from utils import *
import numpy as np
import torch
import os,sys
import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

import warnings
warnings.filterwarnings("ignore")

# smiles规范化处理
from rdkit import Chem
from rdkit.Chem import AllChem
import pandas as pd
from collections import defaultdict
from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np
from math import log10
from rdkit import DataStructs
from rdkit.Chem import MolStandardize
#from neutral import NeutraliseCharges
from multiprocessing import Pool
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

def _InitialiseNeutralisationReactions():
    patts= (
    # Imidazoles
    ('[n+;H]','n'),
    # Amines
    ('[N+;!H0]','N'),
    # Carboxylic acids and alcohols
    ('[$([O-]);!$([O-][#7])]','O'),
    # Thiols
    ('[S-;X1]','S'),
    # Sulfonamides
    ('[$([N-;X2]S(=O)=O)]','N'),
    # Enamines
    ('[$([N-;X2][C,N]=C)]','N'),
    # Tetrazoles
    ('[n-]','[nH]'),
    # Sulfoxides
    ('[$([S-]=O)]','S'),
    # Amides
    ('[$([N-]C=O)]','N'),
    )
    return [(Chem.MolFromSmarts(x),Chem.MolFromSmiles(y,False)) for x,y in patts]
_reactions=None
def NeutraliseCharges(smiles, reactions=None):
    global _reactions
    if reactions is None:
        if _reactions is None:
            _reactions=_InitialiseNeutralisationReactions()
        reactions=_reactions
    mol = Chem.MolFromSmiles(smiles)
    replaced = False
    for i,(reactant, product) in enumerate(reactions):
        while mol.HasSubstructMatch(reactant):
            replaced = True
            rms = AllChem.ReplaceSubstructs(mol, reactant, product)
            mol = rms[0]
    if replaced:
        return (Chem.MolToSmiles(mol,True), True)
    else:
        return (smiles, False)


def canonicalize_smiles(smiles):
    if len(smiles)==0:
        return ''
    mol = Chem.MolFromSmiles(smiles)
    lfc = MolStandardize.fragment.LargestFragmentChooser()
    
    if mol is not None:
        mol2 = lfc.choose(mol)
        smi2=Chem.MolToSmiles(mol2, isomericSmiles=True)
        smi,_=NeutraliseCharges(smi2)
        return smi
    else:
        return ''

all_cps=list()
def run(line):
    
    hang=line.rstrip('\n').split(',')
    smi=hang[0]
    smi=canonicalize_smiles(smi)
    m=Chem.MolFromSmiles(smi)
    if smi=='' or m is None:
        return ''
    else:
        return smi


def main(args, drug_smiles_list):

    drug_predict_result = defaultdict(dict)
    for dataset in ['GPCR','IonChannel','Enzyme','Kinase','NHR','Transporter','Other']: #'GPCR','IonChannel','Enzyme','Kinase','NHR','Transporter','Other'
        trained_cpi_model_path = f'/home/datahouse1/liujin/PreMOTA/regression_multitask/model_fintune_save/{dataset}/ratio_0.9batch128LR_1e-4random_0_esm2.pt'

        USE_CUDA = torch.cuda.is_available()
        device = torch.device(args.device if USE_CUDA else 'cpu')
        print(device)


        embedding_path = "/home/datahouse1/liujin/PreMOTA/dataset_reg_multitask/%s/%s/"%(dataset,args.protein_embedding)
        # Obtain the uniprot_id of the target under a certain type of target
        protein_dict = eval(open("/home/datahouse1/liujin/PreMOTA/dataset_reg_multitask/%s/proteins.txt"%dataset,'r').read())
        uniprot_id_list = list(protein_dict.keys())
        print("{} dataset has {} proteins".format(dataset,len(uniprot_id_list)))

        model = CPI_classification(device,
                                emb_size=args.emb_size,
                                max_length=2000,
                                dropout=args.dropout_global,
                                modulator_emb_dim=args.modulator_emb_dim,
                                ppi_emb_dim=args.ppi_emb_dim,
                                h_dim=args.h_dim,
                                n_heads=args.n_heads,
                                n_output=1)
        model.to(device)
        
        checkpoint = torch.load(trained_cpi_model_path, map_location=device)
        model.load_state_dict(checkpoint['net'])
        print(f'predicting on model:{trained_cpi_model_path};device:{device};')
        
        for uniprot_id in tqdm(uniprot_id_list):
        
            batch_smiles = drug_smiles_list
            affinity_value = data_predict_smilesbatch(model, device, batch_smiles, uniprot_id, embedding_path)
            for drug_smiles,affinity_value_item in zip(batch_smiles,affinity_value):
                affinity_value_item = affinity_value_item.tolist() # The former one is the pK value, and the latter one is the pAC50 value
                drug_predict_result[drug_smiles][uniprot_id] = affinity_value_item #{"smiles1":{"uniprot_id1":affinity_value1, "uniprot_id2":affinity_value2}}
                
    return drug_predict_result


            

#['GPCR','IonChannel','Enzyme','Kinase','NHR','Transporter','Other']
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device',default='cuda:5',type=str,help='device id (0,1,2,3)')
    parser.add_argument('--train_batch_size', type=int, default=64)
    parser.add_argument('--test_batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--num_epochs', type=int, default=100) #100
    parser.add_argument('--seed', type=int, default=2024)
    parser.add_argument('--dropout_global', type=float, default=0)
    parser.add_argument('--train_val_ratio', type=float, default=0.9)
    parser.add_argument('--early_stop', type=int, default=25) #10
    parser.add_argument('--stop_epoch', type=int, default=0)
    parser.add_argument('--best_epoch', type=int, default=-1)
    parser.add_argument('--best_pearson', type=float, default=0)
    parser.add_argument('--last_epoch', type=int, default=1)
    parser.add_argument('--best_model', type=str, default=None)
    parser.add_argument('--modulator_emb_dim', type=int, default=256)
    parser.add_argument('--ppi_emb_dim', type=int, default=512)
    parser.add_argument('--h_dim', type=int, default=512)
    parser.add_argument('--n_heads', type=int, default=2)
    parser.add_argument('--protein_embedding', type=str, default='esm2',choices=['onehot','tape', 'esm1b', 'esm2','esm2_1024'])
    parser.add_argument('--emb_size', type=int, default=1280,choices=[20, 768, 1280, 1280])
    args = parser.parse_args()
    

    data_path = '/home/datahouse1/liujin/PreMOTA/dataset_predict/investigational_drugs/'
    data_df1 = pd.read_excel(os.path.join(data_path,"drugs_info.xls"),sheet_name="Investigational_Drugs")
    data_df2 = pd.read_excel(os.path.join(data_path,"drugs_info.xls"),sheet_name="Withdrawn_Drugs")
    # Select the drug_name, smiles, and clinial_trials columns in data_df1
    data_df_1 = data_df1[["drug_name","smiles","clinial_trials"]]
    # Modify the column name clinical_trials to toxtype
    data_df_1 = data_df_1.rename(columns={"clinial_trials":"toxtype"})
    data_df_2 = data_df2[["drug_name","smiles","toxtype"]]
    print("data_df_1 shape is:",data_df_1.shape)
    print("data_df_2 shape is:",data_df_2.shape)
    data_df = pd.concat([data_df_1,data_df_2],axis=0)
    data_df = data_df.drop_duplicates(subset=['drug_name'])

    print("before prase data_df shape is:",data_df.shape)
    # data_df = data_df2[["drug_name","smiles","clinical_trials"]]
    print("Prasing data start!")
    data_df["smiles"] = data_df["smiles"].astype(str)
    data_df["canonical_smiles"] = data_df["smiles"].apply(lambda x: run(x))
    data_df = data_df.dropna(subset=['canonical_smiles'])
    data_df = data_df.drop_duplicates(subset=['canonical_smiles'])
    data_df = data_df.reset_index(drop=True)
    print("after prase data_df shape is:",data_df.shape)
    data_df.to_csv(os.path.join(data_path,"drugs_info.csv"),index=False)
    print("Prasing data end!")
    all_smiles_list = data_df["canonical_smiles"].tolist()
    print("all smiles length is:",len(all_smiles_list))

    drug_predict_result = main(args, all_smiles_list)


    # Save the predicted results to a JSON file
    result_save_path = os.path.join(data_path, "drugs_offtarget_predict_result.json")
    with open(result_save_path,'w') as f:
        json.dump(drug_predict_result,f)


# nohup python -u predict_drugs.py >investigational_drugs_predict.log 2>&1 &
