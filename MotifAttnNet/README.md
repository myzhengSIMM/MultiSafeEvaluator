## Usage


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

