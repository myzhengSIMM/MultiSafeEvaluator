## Usage


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