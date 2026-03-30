#!/bin/bash

python3.11 -m venv environment
source environment/bin/activate
pip install -r requirements.txt

python3 train.py --gsl_type type_aware --k_additional 1 --n_epochs 20
python3 train.py --gsl_type type_aware --k_additional 3 --n_epochs 20
python3 train.py --gsl_type type_aware --k_additional 5 --n_epochs 20
python3 train.py --gsl_type type_aware --k_additional 7 --n_epochs 20

tar -cvf result.tar saved_models
