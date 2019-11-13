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
from lightcurve_visualizer import plot_lightcurve
from tqdm import tqdm
from typing import Union

def inference_postprocessing(label: Union[tf.Tensor, np.ndarray], prediction: Union[tf.Tensor, np.ndarray],
                             length: int) -> (np.ndarray, np.ndarray):
    """
    Prepares the label and prediction for use alongside the original data. In particular, as the network may
    require a specific multiple size, the label and prediction may need to be slightly clipped or padded. Also
    ensures NumPy types for easy use.

    :param label: The ground truth label (preprocessed for use by the network).
    :param prediction: The prediction from the network.
    :param length: The length of the original example before preprocessing.
    :return: The label and prediction prepared for comparison to the original unpreprocessed example.
    """
    if isinstance(label, tf.Tensor):
        label = label.numpy()
    if isinstance(prediction, tf.Tensor):
        prediction = prediction.numpy()
    if label.shape[0] > length:
        label = label[:length]
        prediction = prediction[:length]
    elif label.shape[0] < length:
        elements_to_repeat = length - label.shape[0]
        label = np.pad(label, (0, elements_to_repeat), mode='constant')
        prediction = np.pad(prediction, (0, elements_to_repeat), mode='constant')
    return label, prediction

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
for example_path in tqdm(example_paths):
    example, label = database.evaluation_preprocessing(tf.convert_to_tensor(example_path))
    prediction = model.predict(tf.expand_dims(example, axis=0))[0]
    lightcurve_data_frame = pd.read_feather(example_path)  # Not required for prediction, but useful for analysis.
    # Use prediction here as desired.
    fluxes = lightcurve_data_frame['flux'].values
    times = lightcurve_data_frame['HJD'].values
    label, prediction = inference_postprocessing(label, prediction, fluxes.shape[0])
    thresholded_prediction = prediction > 0.5  # Can threshold on some probability.
    # Can plot thresholded_predictions and fluxes here.
    plot_lightcurve(times, fluxes, label, prediction, title=example_path, save_path=f'inference_plots/{example_path}.png')
