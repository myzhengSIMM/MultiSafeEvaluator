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
conda create -n premota python=3.8
conda activate premota
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
pip install -r requirements.txt
```

### Training

Run `run_cmax_dose_random.sh` and `run_cmax_dose_scaffold.sh`, save the metric results, and store the models in the ./result/Cmax/ directory for different seed values.

Run `run_ppb_random.sh` and `run_ppb_scaffold.sh`, save the metric results, and store the models in the ./result/PPB/ directory for different seed values.


### Predict

Run `predict_drugs_dose.py` to obtain the PPB, Cmax, and Cmax,free values of the compounds.

