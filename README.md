# PfAbNet-viscosity
This repository accompanies the manuscript "Low-Data Interpretable Deep Learning Prediction of Antibody Viscosity using a Biophysically Meaningful Representation." The code and notebooks in this repository can be used to train PfAbNet-viscosity models, generate test set predictions and reproduce all analyses reported in the manuscript.
![alt text](https://github.com/PfizerRD/PfAbNet-viscosity/blob/main/images/PfAbNet-viscosity_workflow.png?raw=true)

This workflow requires the following software/toolkits licenses:
1. Bioluminate (Schrodinger LLC)
2. oechem, oespicoli, and oezap toolkits (OpenEye Scientific Software)

Run the jupyter notebooks in the following order to reproduce the analyses presented in the manuscript:
1. ```1_preprocess.ipynb```: Retrieve and process the raw data (measured viscosity and antibody sequences)
2. ```2_build_hm.ipynb```: Build homology models. Analyze and plot dataset diversity.
3. ```3_validation.ipynb```: Train PfAbNet-PDGF and PfAbNet-Ab21 models. Generate test set predictions and performance plots.
4. ```4_attribution.ipynb```: Perform attribution analysis.
5. ```5_sensitivity.ipynb```: Perform sensitivity analysis.

## Training
The following command can be used to train PfAbNet models from the command line after the required input files have been created (see the Jupyter Notebooks on how to specify input arguments).

For example, this will train models using PDGF38 dataset.

```
python pfabnet/train.py --training_data_file data/PDGF.csv \
  --homology_model_dir data/hm \
  --output_model_prefix PfAbNet-PDGF38 \
  --output_model_dir models/pdgf38
```

## Inference
The following command can be used to generate predictions for a test antibody using .mol2 file with charges (see the Jupyter Notebooks on how to specify input arguments).

```
python pfabnet/predict.py --structure_file data/hm/mAb1.mol2 \
  --PfAbNet_model_dir models/pdgf38 \
  --PfAbNet_model_prefix PfAbNet-PDGF38 \
  --output_file models/pdgf/mAb1.csv
```

