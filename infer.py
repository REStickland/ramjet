"""Code for inference on the contents of a directory."""
#Tells the  infer script not to use the GPU:
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
#follows normally
import numpy as np
import pandas as pd
import tensorflow as tf
from pathlib import Path

from models import ConvolutionalLstm
from photometric_database.microlensing_label_per_time_step_database import MicrolensingLabelPerTimeStepDatabase

# Set these paths to the correct paths.
saved_log_directory = Path('logs/convolutional LSTM 2019-10-01-18-27-13')
meta_data_path = Path('data/candlist_RADec.dat.feather')

print('Setting up dataset...')
database = MicrolensingLabelPerTimeStepDatabase()
database.meta_data_frame = pd.read_feather(meta_data_path)
example_paths = pd.read_csv(saved_log_directory.joinpath('validation.csv'), header=None)[0].values

print('Loading model...')
model = ConvolutionalLstm()
model.load_weights(str(saved_log_directory.joinpath('model.ckpt')))

print('Inferring...')
for example_path in example_paths:
    example, label = database.evaluation_preprocessing(tf.convert_to_tensor(example_path))
    prediction = model.predict(tf.expand_dims(example, axis=0))[0]
    lightcurve_data_frame = pd.read_feather(example_path)  # Not required for prediction, but useful for analysis.
    # Use prediction here as desired.
