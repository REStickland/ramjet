"""Code for inference on the contents of a directory."""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import datetime
import io
import random

import numpy as np
import pandas as pd
import tensorflow as tf
from pathlib import Path

from ramjet.analysis.lightcurve_visualizer import plot_lightcurve
from ramjet.analysis.model_loader import get_latest_log_directory
from ramjet.models import ConvolutionalLstm, SimpleLightcurveCnn
from ramjet.photometric_database.self_lensing_binary_per_example_database import SelfLensingBinaryPerExampleDatabase
from ramjet.photometric_database.toi_lightcurve_database import ToiLightcurveDatabase

# saved_log_directory = get_latest_log_directory('logs')  # Uses the latest log directory's model.
saved_log_directory = Path('logs/SLB, all signals training, no less than 2 day period, all tess lc, PDCSAP 2020-01-21-14-08-39')

print('Setting up dataset...')
database = SelfLensingBinaryPerExampleDatabase()
example_paths = Path('/local/data/fugu3/sishitan/TESSdata/lcurve').glob('**/*.fits')
example_paths = list(example_paths)
random.shuffle(example_paths)
# Uncomment below to run the inference for all validation files.
# example_paths = pd.read_csv(saved_log_directory.joinpath('validation.csv'), header=None)[0].values

datetime_string = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

print('Loading model...')
model = SimpleLightcurveCnn()
model.load_weights(str(saved_log_directory.joinpath('model.ckpt')))

print('Inferring and plotting...')
columns = ['Lightcurve path', 'Prediction']
dtypes = [str, int]
predictions_data_frame = pd.read_csv(io.StringIO(''), names=columns, dtype=dict(zip(columns, dtypes)))
old_top_predictions_data_frame = predictions_data_frame
for index, example_path in enumerate(example_paths):
    example = database.infer_preprocessing(example_path)
    prediction = model.predict(tf.expand_dims(example, axis=0))[0][0]
    predictions_data_frame = predictions_data_frame.append({'Lightcurve path': str(example_path),
                                                            'Prediction': prediction},
                                                           ignore_index=True)
    if index % 1000 == 0:
        print(index)
        predictions_data_frame.sort_values('Prediction', ascending=False).reset_index().to_feather(
            f'another pdcsap predictions {datetime_string}.feather'
        )
predictions_data_frame.to_feather(f'predictions {datetime_string}.feather')
