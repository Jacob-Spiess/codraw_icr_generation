#!/bin/bash

export PYTHONPATH=.:$PYTHONPATH

conda create --name codraw_pl python=3.10
conda activate codraw_pl
# from https://pytorch.org/get-started/previous-versions/
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.6 -c pytorch -c nvidia
conda install lightning -c conda-forge
conda install h5py
conda install pandas
conda install seaborn
# torchmetrics is not installing v1.0
conda install -c conda-forge torchmetrics
conda install -c huggingface transformers
conda install -c anaconda -c conda-forge -c comet_ml comet_ml
conda install scipy
# update lightning to 2.0.8, installed but did not update?
# conda install lightning -c conda-forge
pip install positional-encodings[pytorch]
conda install -c conda-forge scikit-learn
pip install typing-extensions
pip install sympy
pip install numpy
pip install Pillow
python -m pip install requests
pip install six
pip install -U 'jsonschema<4.0'
python -m pip install psutil
pip install pytorch-lightning
conda install protobuf
pip install absl-py