import json
import pandas as pd
import torch
import numpy as np
import os
import random
# Utils
from utils.utils import DataLoader, compute_pna_degrees, virtual_screening, CustomWeightedRandomSampler, eval_test
from utils.dataset import *  # data
from utils.trainer import Trainer
from utils.metrics import *
# Preprocessing
from utils import ligand_init
import optuna
# Model
from models.net import net
import argparse
import ast
import warnings
warnings.filterwarnings("ignore")


### Data and Pre-processing
datafolder = '/home/datahouse1/liujin/project_simm/MotifAttnNet/datasets/PPB/ppb_scaffold_split'
result_path = '/home/datahouse1/liujin/project_simm/MotifAttnNet/result/PPB/ppb_scaffold_split_optuna_SetRep_nomotif/'


device = torch.device('cuda:6')
seed = 2024


# device
if not os.path.exists(result_path):
    os.makedirs(result_path)

model_path = os.path.join(result_path,'save_model_seed{}'.format(seed))
if not os.path.exists(model_path):
    os.makedirs(model_path)



# with open(os.path.join(result_path, 'model_params.txt'), 'w') as f:
#     f.write(str(args))

## import files
train_df = pd.read_csv(os.path.join(datafolder,'train.csv'))
test_df = pd.read_csv(os.path.join(datafolder,'test.csv'))

valid_path = os.path.join(datafolder,'valid.csv')
valid_df = 1 #None
if os.path.exists(valid_path):
    valid_df = pd.read_csv(valid_path)
    ligand_smiles = list(set(train_df['smiles'].tolist()+test_df['smiles'].tolist()+valid_df['smiles'].tolist())) 
else:
    ligand_smiles = list(set(train_df['smiles'].tolist()+test_df['smiles'].tolist())) 


ligand_path = os.path.join(datafolder,'ligand.pt')
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
## training loader
train_shuffle = True
train_sampler = None



train_dataset = MotifMoleculeDataset(train_df, ligand_dict, device=device)
test_dataset = MotifMoleculeDataset(test_df, ligand_dict, device=device)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=train_shuffle,
                        sampler=train_sampler, follow_batch=['mol_x', 'clique_x']) 

test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False,
                            follow_batch=['mol_x', 'clique_x']) 


valid_dataset, valid_loader = None, None
if valid_df is not None:
    valid_dataset = MotifMoleculeDataset(valid_df, ligand_dict, device=device)
    valid_loader = DataLoader(valid_dataset, batch_size=64, shuffle=False,
                                follow_batch=['mol_x', 'clique_x']
                                )
    

degree_path = os.path.join(result_path,'save_model_seed{}'.format(seed),'degree.pt')
if not os.path.exists(degree_path):
    print('Computing training data degrees for PNA')
    mol_deg, clique_deg = compute_pna_degrees(train_loader)
    degree_dict = {'ligand_deg':mol_deg, 'clique_deg':clique_deg}
    torch.save(degree_dict, degree_path)
else:
    degree_dict = torch.load(degree_path)
    mol_deg, clique_deg = degree_dict['ligand_deg'], degree_dict['clique_deg']

def objective(trial):
    config = {
        'params':{
            'hidden_channels': trial.suggest_categorical('hidden_channels', [100, 200, 300, 400, 500]),
            'num_layers': trial.suggest_categorical('num_layers', [2, 3, 4, 5]),
            "clique_num_timesteps": trial.suggest_categorical('clique_num_timesteps', [1, 2, 3, 4]),
            'num_timesteps': trial.suggest_categorical('num_timesteps', [1, 2, 3, 4]),
            # 'hidden_channels': trial.suggest_categorical('hidden_channels', [200,]),
            # 'num_layers': trial.suggest_categorical('num_layers', [3,]),
            # 'num_timesteps': trial.suggest_categorical('num_timesteps', [2,]),
            'n_hidden_sets': trial.suggest_categorical('n_hidden_sets', [64, 128, 256, 512]),
            'n_elements': trial.suggest_categorical('n_elements', [8, 32, 64, 128]),
            'total_layers': trial.suggest_categorical('total_layers', [1,2,3]),
            'dropout': trial.suggest_categorical('dropout', [0.0,])
            # 'dropout': trial.suggest_categorical('dropout', [0.0, 0.1, 0.2, 0.3, 0.4, 0.5])
        },
        'tasks':{
                
            'regression_task': trial.suggest_categorical('regression_task', [True]),
            'classification_task': trial.suggest_categorical('classification_task', [False]),
            'mclassification_task': trial.suggest_categorical('mclassification_task', [False]),
        },
        'optimizer':{
            'lrate': trial.suggest_categorical('lrate', [1e-3, 1e-4, 1e-5, 1e-5]),
            'weight_decay': trial.suggest_categorical('weight_decay', [1e-3, 1e-4, 1e-5, 1e-6, 1e-7]),
            'eps': trial.suggest_categorical('eps', [1e-8, 1e-8, 1e-7, 1e-6, 1e-5]),
            'betas': trial.suggest_categorical('betas', [(0.9, 0.999), (0.8, 0.999)])
        }
        
            }
    device = torch.device('cuda:4')
    seed = 2024
    # seed initialize
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

    model = net(mol_deg,
                # MOLECULE
                mol_in_channels=43, 
                hidden_channels=config['params']['hidden_channels'],                 
                dropout=config['params']['dropout'],
                # dropout_attn_score=config['params']['dropout_attn_score'],
                # drop_atom=config['params']['drop_atom'],
                num_layers=config['params']['num_layers'],
                clique_num_timesteps=config['params']['clique_num_timesteps'],
                # num_timesteps=config['params']['num_timesteps'],
                n_hidden_sets=config['params']['n_hidden_sets'],
                n_elements=config['params']['n_elements'],
                # output
                regression_head=config['tasks']['regression_task'],
                classification_head=config['tasks']['classification_task'] ,
                multiclassification_head=config['tasks']['mclassification_task'],
                set_layer="SetRep", # deepset SetRep
                device=device).to(device)

    evaluation_metric = 'rmse' #pearson rmse
    

    engine = Trainer(model=model, lrate=config['optimizer']['lrate'], min_lrate=0,
                        wdecay=config['optimizer']['weight_decay'], betas=config['optimizer']['betas'], 
                        eps=config['optimizer']['eps'], amsgrad=False,
                        clip=1, steps_per_epoch=len(train_loader), 
                        num_epochs=50,total_iters = None, 
                        warmup_iters=0, 
                        lr_decay_iters=0,
                        schedule_lr=False, regression_weight=1, classification_weight=1, 
                        evaluate_metric=evaluation_metric, result_path=result_path, runid=seed, 
                        device=device)

    print('-'*50)
    print('start training model')

    best_value = engine.train_epoch(train_loader, val_loader = valid_loader, test_loader = test_loader, evaluate_epoch = 1)
    return best_value

study = optuna.create_study(direction='minimize')
# study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=200)
best_params = study.best_params
best_value = study.best_value

print('Best trial:')
print('Value: ', best_value)

for key, value in best_params.items():
    print('{}: {}'.format(key, value))

# save the best params
with open(os.path.join(result_path,'best_params_seed{}.json'.format(seed)), 'w') as f:
    json.dump(best_params, f)
