import json
import pandas as pd
import torch
import numpy as np
import os
import random
# Utils
from utils.utils import DataLoader, compute_pna_degrees, eval_predict_ppb_cmax
from utils.dataset import *  # data
from utils.dataset_dose import *
from utils.trainer import Trainer
from utils.metrics import *
# Preprocessing
from utils import ligand_init
# Model
from models.net import net
import argparse
import ast
import warnings
warnings.filterwarnings("ignore")

def tuple_type(s):
    try:
        # Safely evaluate the string as a tuple
        value = ast.literal_eval(s)
        if not isinstance(value, tuple):
            raise ValueError
    except (ValueError, SyntaxError):
        raise argparse.ArgumentTypeError(f"Invalid tuple value: {s}")
    return value

def list_type(s):
    try:
        # Safely evaluate the string as a tuple
        value = ast.literal_eval(s)
        if not isinstance(value, list):
            raise ValueError
    except (ValueError, SyntaxError):
        raise argparse.ArgumentTypeError(f"Invalid list value: {s}")
    return value

parser = argparse.ArgumentParser()

### Seed and device
parser.add_argument('--seed', type=int, default=2024)
parser.add_argument('--device', type=str, default='cuda:5', help='')
parser.add_argument('--config_path',type=str,default='/home/datahouse1/liujin/project_simm/MotifAttnNet/config.json')
### Data and Pre-processing
parser.add_argument('--datafolder', type=str, default='/home/datahouse1/liujin/project_simm/MotifAttnNet/datasets/ADR_tree_label/', help='model data path')
parser.add_argument('--result_path', type=str,default='/home/datahouse1/liujin/project_simm/MotifAttnNet/datasets/ADR_tree_label/dose_study/',help='path to save results')
parser.add_argument('--save_interpret', type=bool,default=False,help='path to save results')

# For PDBBIND datasets - we train for 30K iteration 
parser.add_argument('--regression_task',type=bool, help='True if regression else False',default=True) 
# For any classification type - we train for 100 epochs (same as DrugBAN) [change --total_iters = None]
parser.add_argument('--classification_task',type=bool, help='True if classification else False') 
parser.add_argument('--mclassification_task',type=int, help='number of multiclassification, 0 if no multiclass task')
parser.add_argument('--epochs', type=int, default=1, help='')
parser.add_argument('--evaluate_epoch',type=int,default=1)

parser.add_argument('--total_iters',type=int,default=None) 
parser.add_argument('--evaluate_step',type=int,default=500)


# optimizer params - only change this for PDBBind v2016
parser.add_argument('--lrate',type=float,default=1e-4,help='learning rate for PSICHIC') # change to 1e-5 for LargeScaleInteractionDataset
parser.add_argument('--eps',type=float,default=1e-8, help='higher = closer to SGD') # change to 1e-5 for PDBv2016
parser.add_argument('--betas',type=tuple_type, default="(0.9,0.999)")  # change to (0.9,0.99) for PDBv2016
# batch size
parser.add_argument('--batch_size',type=int,default=128)
parser.add_argument('--set_layer', type=str, default="SetRep")
# sampling method - only used for pretraining large-scale interaction dataset ; allow self specified weights to the samples



args = parser.parse_args()

with open(args.config_path,'r') as f:
        config = json.load(f)

# overwrite 
config['optimizer']['lrate'] = args.lrate
config['optimizer']['eps'] = args.eps
config['optimizer']['betas'] = args.betas
config['tasks']['regression_task'] = args.regression_task
config['tasks']['classification_task'] = args.classification_task
config['tasks']['mclassification_task'] = args.mclassification_task
config['params']['set_layer'] = args.set_layer


# device
device = torch.device(args.device)

import pandas as pd
import numpy as np

from rdkit import Chem
from rdkit.Chem import Descriptors
import re

def calculate_molecular_weight(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        molecular_weight = Descriptors.MolWt(mol)
    except:
        molecular_weight = np.nan
    return molecular_weight


## import files
df_filter_all = pd.read_csv(os.path.join(args.datafolder,'adrdrug_dup_label_filter_dropdup.csv'))
ligand_smiles = list(set(df_filter_all['smiles'].tolist()))


ligand_path = os.path.join(args.datafolder,'ligand.pt')
if os.path.exists(ligand_path):
    print('Loading Ligand Graph data...')
    ligand_dict = torch.load(ligand_path)
else:
    print('Initialising Ligand SMILES to Ligand Graph...')
    ligand_dict = ligand_init(ligand_smiles)
    torch.save(ligand_dict,ligand_path)

torch.cuda.empty_cache()
##TODO: drop any invalid smiles

##TODO: drop any invalid smiles

## 

predict_dataset = MotifMoleculeDataset(df_filter_all, ligand_dict, device=args.device)


def predict_cmax(predict_df,ppb_model_param_dict,cmax_dose_model_param_dict):
   
    model_ppb = net(mol_deg=None,
                # MOLECULE
                mol_in_channels=config['params']['mol_in_channels'], 
                hidden_channels=config['params']['hidden_channels'],                 
                num_layers=config['params']['num_layers'],
                # heads=config['params']['heads'],
                clique_num_timesteps=config['params']['clique_num_timesteps'],
                num_timesteps=config['params']['num_timesteps'],
                n_hidden_sets=config['params']['n_hidden_sets'],
                n_elements=config['params']['n_elements'],
                dropout=config['params']['dropout'],
                # dropout_attn_score=config['params']['dropout_attn_score'],
                # output
                regression_head=config['tasks']['regression_task'],
                classification_head=config['tasks']['classification_task'] ,
                multiclassification_head=config['tasks']['mclassification_task'],
                set_layer=config['params']['set_layer'],
                dose_mode=False,
                device=device).to(device)

    model_cmax = net(mol_deg=None,
                # MOLECULE
                mol_in_channels=config['params']['mol_in_channels'], 
                hidden_channels=config['params']['hidden_channels'],                 
                num_layers=config['params']['num_layers'],
                # heads=config['params']['heads'],
                clique_num_timesteps=config['params']['clique_num_timesteps'],
                num_timesteps=config['params']['num_timesteps'],
                n_hidden_sets=config['params']['n_hidden_sets'],
                n_elements=config['params']['n_elements'],
                dropout=config['params']['dropout'],
                # dropout_attn_score=config['params']['dropout_attn_score'],
                # output
                regression_head=config['tasks']['regression_task'],
                classification_head=config['tasks']['classification_task'] ,
                multiclassification_head=config['tasks']['mclassification_task'],
                set_layer=config['params']['set_layer'],
                dose_mode=True,
                device=device).to(device)
    model_ppb.reset_parameters() #重置模型中的所有参数，以便在训练过程中重新初始化模型参数
    model_cmax.reset_parameters()

    model_ppb.load_state_dict(torch.load(ppb_model_param_dict, map_location=args.device),strict=False)
    model_cmax.load_state_dict(torch.load(cmax_dose_model_param_dict, map_location=args.device),strict=False)


    print('Pretrained model loaded!')
   
    print('loading saved checkpoint and predicting data')

    df = eval_predict_ppb_cmax(screen_df=predict_df, model_ppb=model_ppb, model_cmax=model_cmax, 
                                        data_loader=predict_loader, data_loader_dose=predict_loader_dose, device=args.device) #, dose_mode=args.dose_mode

    # 新增predicted_activity_ppb列和predicted_activity_cmax列
    df['ppb'] = (1-10**df['predicted_activity_ppb'])/0.999
    df['cmax(ug/ml)'] = 10**df['predicted_activity_cmax']
    df['cmax_free(ug/ml)'] = (1- df['ppb'])*df['cmax(ug/ml)']
    df['molecular_weight'] = df['smiles'].apply(calculate_molecular_weight) #canonical_smiles
    df['cmax_free(uM)'] = df['cmax_free(ug/ml)']/df['molecular_weight'] * 1000

    del model_ppb
    del model_cmax
    torch.cuda.empty_cache()

    
    return df.copy()

dose_list = [10, 100,] #[5, 10, 25, 50, 75, 100, 125, 150, 200, 250,300,350]

for dose in dose_list:
    print('Now dose is:',dose)
    df_filter_all2 = df_filter_all.copy()
    df_filter_all2['dose'] = dose

    predict_dataset_dose = MotifMoleculeDataset_dose(df_filter_all2, ligand_dict, device=args.device)
        

    predict_loader = DataLoader(predict_dataset, batch_size=args.batch_size, shuffle=False,
                                follow_batch=['mol_x', 'clique_x'])
    predict_loader_dose = DataLoader(predict_dataset_dose, batch_size=args.batch_size, shuffle=False,
                                follow_batch=['mol_x', 'clique_x'])
    
    # screen_df1 = predict_cmax(df_filter_all2,"/home/datahouse1/liujin/project_simm/MotifAttnNet_rep3/result/PPB/PPB_motif_setrep/ppb_scaffold_split_run2024/save_model_seed2024/model.pt","/home/datahouse1/liujin/project_simm/MotifAttnNet_rep3/result/Cmax/cmax_random_split_run1998/save_model_seed1998/model.pt")
    # screen_df2 = predict_cmax(df_filter_all2,"/home/datahouse1/liujin/project_simm/MotifAttnNet_rep3/result/PPB/PPB_motif_setrep/ppb_scaffold_split_run2024/save_model_seed2024/model.pt","/home/datahouse1/liujin/project_simm/MotifAttnNet_rep3/result/Cmax/cmax_random_split_run2022/save_model_seed2022/model.pt")
    # screen_df3 = predict_cmax(df_filter_all2,"/home/datahouse1/liujin/project_simm/MotifAttnNet_rep3/result/PPB/PPB_motif_setrep/ppb_scaffold_split_run2024/save_model_seed2024/model.pt","/home/datahouse1/liujin/project_simm/MotifAttnNet_rep3/result/Cmax/cmax_random_split_run2023/save_model_seed2023/model.pt")
    # screen_df4 = predict_cmax(df_filter_all2,"/home/datahouse1/liujin/project_simm/MotifAttnNet_rep3/result/PPB/PPB_motif_setrep/ppb_scaffold_split_run2024/save_model_seed2024/model.pt","/home/datahouse1/liujin/project_simm/MotifAttnNet_rep3/result/Cmax/cmax_random_split_run2024/save_model_seed2024/model.pt")
    # screen_df5 = predict_cmax(df_filter_all2,"/home/datahouse1/liujin/project_simm/MotifAttnNet_rep3/result/PPB/PPB_motif_setrep/ppb_scaffold_split_run2024/save_model_seed2024/model.pt","/home/datahouse1/liujin/project_simm/MotifAttnNet_rep3/result/Cmax/cmax_random_split_run2025/save_model_seed2025/model.pt")
    # dfs = [screen_df1, screen_df2, screen_df3, screen_df4, screen_df5]
    # numeric_cols = ["predicted_activity_ppb","predicted_activity_cmax","ppb","cmax(ug/ml)","cmax_free(ug/ml)","molecular_weight","cmax_free(uM)"]
    
    ## Calculate the sum of the numerical columns and then compute the mean
    # sum_df = sum(df[numeric_cols] for df in dfs)  #Compute the sum column by column
    # avg_df = sum_df / len(dfs)
    ## Copy the first DataFrame and replace the numerical columns with the new mean values
    # screen_df = dfs[0].copy()
    # screen_df[numeric_cols] = avg_df
    

    screen_df = predict_cmax(df_filter_all2,
                            "/home/datahouse1/liujin/project_simm/MotifAttnNet/result/PPB/PPB_motif_setrep/ppb_scaffold_split_run2024/save_model_seed2024/model.pt",
                          "/home/datahouse1/liujin/project_simm/MotifAttnNet/result/Cmax/cmax_random_split_run2024/save_model_seed2024/model.pt")
    screen_df.to_csv(os.path.join(args.result_path,f'drug_cmax_predict_{dose}mg.csv'),index=False)