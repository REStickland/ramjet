"""Code for inference on the contents of a directory."""
import pandas as pd
import tensorflow as tf
from pathlib import Path

from ramjet.analysis.lightcurve_visualizer import plot_lightcurve
from ramjet.analysis.model_loader import get_latest_log_directory
from ramjet.models import ConvolutionalLstm, SimpleLightcurveCnn
from ramjet.photometric_database.microlensing_label_per_example_database import MicrolensingLabelPerExampleDatabase
from ramjet.photometric_database.toi_lightcurve_database import ToiLightcurveDatabase

# saved_log_directory = get_latest_log_directory('logs')  # Uses the latest log directory's model.
saved_log_directory = Path('logs/MOA microlensing, label per example, simple CNN 2019-12-11-23-41-01')

print('Setting up dataset...')
database = MicrolensingLabelPerExampleDatabase()
# Uncomment below to run the inference for all validation files.
example_paths = pd.read_csv(saved_log_directory.joinpath('validation.csv'), header=None)[0].values

print('Loading model...')
model = SimpleLightcurveCnn()
model.load_weights(str(saved_log_directory.joinpath('model.ckpt')))

print('Inferring and plotting...')
for example_path in example_paths:
    lightcurve = pd.read_feather(example_path)
    fluxes = lightcurve['flux'].values
    times = lightcurve['HJD'].values
    example = database.preprocess_and_augment_lightcurve(fluxes)
    example_tensor = tf.convert_to_tensor(example)
    prediction = model.predict(tf.expand_dims(example_tensor, axis=0))[0]
    # plot_lightcurve(times, fluxes, title=example_path, save_path=f'{example_path}.png')
