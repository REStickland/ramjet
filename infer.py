"""Code for inference on the contents of a directory."""
#Tells the  infer script not to use the GPU:
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
#follows normally
import tensorflow as tf
from pathlib import Path
import pandas as pd
import matplotlib
#"non-interactive" matplotlib backend
matplotlib.use('Agg')
from ramjet.analysis.lightcurve_visualizer import plot_lightcurve
from ramjet.analysis.model_loader import get_latest_log_directory
from ramjet.models import ConvolutionalLstm
from ramjet.photometric_database.microlensing_label_per_time_step_database import MicrolensingLabelPerTimeStepDatabase
from tqdm import tqdm

tf.get_logger().setLevel('ERROR')

# Set these paths to the correct paths.
meta_data_path = Path('data/moa_microlensing/candlist_RADec.dat.feather')
saved_log_directory = Path('logs/convolutional LSTM 2019-10-01-18-27-13')
#saved_log_directory = get_latest_log_directory('logs')  # Uses the latest log directory's model.
# saved_log_directory = Path('logs/baseline YYYY-MM-DD-hh-mm-ss')  # Specifies a specific log directory's model to use.

print('Setting up dataset...')
database = MicrolensingLabelPerTimeStepDatabase()
database.meta_data_frame = pd.read_feather(meta_data_path)
#database.obtain_meta_data_frame_for_available_lightcurves()
#example_paths = [str(database.lightcurve_directory.joinpath('tess2018319095959-s0005-0000000117979897-0125-s_lc.fits'))]
# Uncomment below to run the inference for all validation files.
example_paths = pd.read_csv(saved_log_directory.joinpath('validation.csv'), header=None)[0].values

print('Loading model...')
model = ConvolutionalLstm()
model.load_weights(str(saved_log_directory.joinpath('model.ckpt')))

print('Inferring and plotting...')
for example_path in tqdm(example_paths):
    example, label = database.evaluation_preprocessing(tf.convert_to_tensor(example_path))
    prediction = model.predict(tf.expand_dims(example, axis=0))[0]
    lightcurve_data_frame = pd.read_feather(example_path)  # Not required for prediction, but useful for analysis.
    # Use prediction here as desired.
    fluxes = lightcurve_data_frame['flux'].values
    times = lightcurve_data_frame['HJD'].values
    label, prediction = database.inference_postprocessing(label, prediction, times.shape[0])
    thresholded_prediction = prediction > 0.5  # Can threshold on some probability.
    # Can plot thresholded_predictions and fluxes here.
    if len(fluxes) >= 50.0:
        if prediction >= 90.0:
            plot_lightcurve(times, fluxes, label, prediction, title=example_path, save_path=f'inference_plots/{example_path}.png')
