#!/bin/bash


pip install -r LSTM-Model/requirements.txt
cd LSTM-Model
python3 model_maker.py
python3 implementation.py
