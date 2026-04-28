# MultiSafeEvaluator
Drug development faces a high attrition rate, mainly due to safety concerns in clinical phases — particularly adverse drug reactions (ADRs) caused by off-target effects. To address this, we propose MultiSafeEvaluator, an innovative multi-dimensional evaluation framework for ADR prediction, integrating drug-off-target affinity and pharmacokinetic (PK) parameters to improve the accuracy of drug safety evaluation.

## Core Modules
- PreMOTA: Off-target affinity prediction module built on a pre-training-fine-tuning strategy.
- MotifAttNet: PK parameter prediction module developed with motif-level attention mechanisms.
- HetSia-SafeNet: Heterogeneous network module integrating drug-off-target features, PK features, and a learnable ADR representation layer.
<img width="815" height="921" alt="image" src="https://github.com/user-attachments/assets/120f7d1f-f41a-49f5-973e-16a92ca50d4d" />

## PreMOTA
## Environment Configuration
```bash
git clone https://github.com/myzhengSIMM/MultiSafeEvaluator.git
conda create -n premota python=3.8
conda activate premota
cd premota
pip install -r requirements.txt
```
## Classification model pre-training
### Pre-training code for the classification model is located in './src'. Usage details follow:
### 1. Prepare your classification data
The data used for pre-training is located in the `./datasets/CPI_data_cls` directory. You can use the provided `CPI_data_cls` dataset or prepare your own dataset. The data can be downloaded from [this link](https://drive.google.com/drive/folders/1ABTd3h1jPA_4PJShuA7SJiSteqb-vOHo?usp=sharing) and should be placed in the `raw_data` folder under `./datasets/CPI_data_cls`.
Run `./src/data/data_process.py` to generate the training and validation datasets for the classification model.

### 2. Train the classification model
Run `./src/train.py` to train the classification model. The trained models will be saved in the `./src/model_save/CPI_data_cls` directory. For each run, the best model is saved as `bach1LR0.0001random2024esm2.pt`. You can use this model for subsequent affinity fine-tuning tasks.
Alternatively, you can download the pre-trained model from [this link](https://drive.google.com/drive/folders/1ABTd3h1jPA_4PJShuA7SJiSteqb-vOHo?usp=sharing) and place it in the `./src/model_save/CPI_data_cls` directory.

**NOTICE : The pre-trained classification model is available for affinity fine-tuning, you can also retrain a new model with custom data**
## Affinity fine-tuning model
### Affinity fine-tuning code is located in './regression_multitask'. Usage details follow:
### 1. Prepare your affinity data
Data used for fine-tuning is located in the `./dataset_reg_multitask/` directory. Different targets data are located in different subdirectories. You can use the provided data or prepare your own data.

### 2. Fine-tune the regression model
To fine-tune the model, run `./regression_multitask/train_reg_fintune.py`. The fine-tuned models will be saved in the `./regression_multitask/model_fintune_save/` directory. For each prediction target, the best model is saved as `ratio_0.9batch128LR_1e-4random_0_esm2.pt` within the respective target folder.
You can also download the pre-trained models from [this link](https://drive.google.com/drive/folders/12XgxmzpDbAO7uuq9cJ5ptndjHe5xMyEV?usp=drive_link) and place them in the `./regression_multitask/model_fintune_save/` directory.
Alternatively, to train the model from scratch, run `./regression_multitask/train_reg_train.py`. The trained models will be saved in the `./regression_multitask/model_train_save/` directory.

## Predict compound-off-target binding affinity
Run `./regression_multitask/predict_drugs.py` or `./regression_multitask/predict_multidata_adr.py` to predict the compound-off-target binding affinity.

## MotifAttNet
### Environment Configuration
```bash
conda create -n motifattnnet python=3.8
conda activate motifattnnet
cd motifattnnet
pip install -r requirements.txt
```
### Training
Run `run_cmax_dose_random.sh` and `run_cmax_dose_scaffold.sh`, save the metric results, and store the models in the ./result/Cmax/ directory for different seed values.
Run `run_ppb_random.sh` and `run_ppb_scaffold.sh`, save the metric results, and store the models in the ./result/PPB/ directory for different seed values.

### Predict
Run `predict_drugs_dose.py` to obtain the PPB, Cmax, and Cmax,free values of the compounds.

## HetSia-SafeNet
## Environment Configuration
```bash
conda create -n bioact python=3.9
conda activate bioact
cd HetSia-SafeNet
pip install -r requirements.txt
```
## Model Training

### Model training consists of two parts: data preprocessing and model training.

### 1. Data Preprocessing
The ADR prediction model (HetSia-SafeNet) is trained using 9202 drugs with 18 ADR labels. The raw ADR dataset is stored in `./data_process/ADR_tree_label`. You can download the ADR_tree_label folder from [this link](https://drive.google.com/drive/folders/1BrL1Gw12G4eamExrf2lmc_fnAXidmjo_?usp=drive_link).
Run `./data_process/drug_ac50_cmax_get.ipynb` to combine off-target data predicted by PreMOTA and Cmax,free values predicted by the MotifAttnNet model. This will generate the required drug features to construct the training dataset.You can download ADR_multitask_dataset, ADR_multitask_dataset_random_split, and ADR_multitask_dataset_scaffold_split from [this link](https://drive.google.com/drive/folders/1jUWwrYmRuj47Ko7ldkTzn213Y5bgaB_Y?usp=drive_link) and place them in the `./Data` folder.

### 2. Data Splitting
Run `./Data/data_split.ipynb` to split the dataset either randomly or using a scaffold-based approach.

### 3. Model Training
Run `./train_adr_property_adremb.py` to train the model and save the trained model in the `./result_split` directory.

property_feature_adr_emb_random: Model trained with randomly split data.
property_feature_adr_emb_scaffold: Model trained with scaffold-based split data.


## Model Prediction and Visualization

### 1. Data Preparation for Prediction (Merging Off-Target and Cmax Data)
Run `./data_process/data_pro_dose_study_test_indrugs.ipynb` to combine off-target data from PreMOTA and Cmax/free values from MotifAttnNet, creating a format that is compatible with the model for prediction.

### 2. Model Prediction and Visualization
Run `./dose_study_drugs_visual.ipynb` to perform predictions and visualize the results.
