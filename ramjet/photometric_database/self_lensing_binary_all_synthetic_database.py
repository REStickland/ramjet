import shutil
import tarfile
from pathlib import Path
import re

import numpy as np
import pandas as pd
import tensorflow as tf

import requests

from ramjet.photometric_database.microlensing_label_per_example_database import MicrolensingLabelPerExampleDatabase


class SelfLensingBinaryAllSyntheticDatabase(MicrolensingLabelPerExampleDatabase):
    def __init__(self, data_directory='data/self_lensing_binary_all_synthetic'):
        super().__init__(data_directory=data_directory)
        self.positive_data_directory = self.data_directory.joinpath('positive')
        self.negative_data_directory = self.data_directory.joinpath('negative')
        self.time_steps_per_example = 20000

    def create_data_directories(self):
        """
        Creates the data directories to be used by the database.
        """
        self.positive_data_directory.mkdir(parents=True, exist_ok=True)
        self.negative_data_directory.mkdir(parents=True, exist_ok=True)

    def clear_data_directory(self):
        """
        Empties the data directory.
        """
        if self.data_directory.exists():
            shutil.rmtree(self.data_directory)
        self.create_data_directories()

    def convert_raw_synthetic_files_to_project_format(self, compressed_data_path, uncompressed_name,
                                                      type_data_directory):
        """
        Takes the compressed synthetic data provided by Agnieszka Cieplak and converts it to a consistent form
        that's apt for the project.
        """
        type_data_directory.mkdir(parents=True, exist_ok=True)
        with tarfile.open(compressed_data_path, "r:gz") as tar_file:
            tar_file.extractall(path=self.data_directory)
            Path(self.data_directory, uncompressed_name).rename(type_data_directory)
        for lightcurve_txt_path in type_data_directory.glob('*'):
            if not re.match(r'(lc_noise_\d+\.out|NADA_\d+\.txt)', lightcurve_txt_path.name):
                continue
            lightcurve = pd.read_csv(lightcurve_txt_path, delim_whitespace=True, skipinitialspace=True,
                                     names=['Time (MJD)', 'Flux', 'Flux error'], comment='#')
            lightcurve_path = lightcurve_txt_path.with_suffix('.feather')
            lightcurve.to_feather(lightcurve_path)
            lightcurve_txt_path.unlink()  # Delete old file.

    def load_and_preprocess_example_file(self, file_path: tf.Tensor) -> (np.ndarray, int):
        """Loads numpy files from the tensor alongside labels."""
        file_path_string = file_path.numpy().decode('utf-8')
        lightcurve = pd.read_feather(file_path_string)['Flux'].values
        lightcurve = self.preprocess_and_augment_lightcurve(lightcurve)
        return lightcurve.astype(np.float32), [self.is_positive(file_path_string)]

    def download_and_prepare_database_with_all_synthetic_negatives(self):
        print('Clearing data directory...')
        self.clear_data_directory()
        print('Downloading synthetic noise...')
        synthetic_noise_url = ('https://files.slack.com/files-pri/TKNTDRH5J-FRTNH7W80/download/' +
                               'syntheticnoisefiles.tar.gz?pub_secret=dca332e375')
        response = requests.get(synthetic_noise_url)
        synthetic_noise_compressed_data_path = Path(self.data_directory, 'SyntheticNoiseFiles.tar.gz')
        with open(synthetic_noise_compressed_data_path, 'wb') as compressed_file:
            compressed_file.write(response.content)
        self.convert_raw_synthetic_files_to_project_format(synthetic_noise_compressed_data_path, 'NADAtxt',
                                                           self.negative_data_directory)

        print('Downloading synthetic signal with noise...')
        synthetic_signals_with_noise_url = ('https://files.slack.com/files-pri/TKNTDRH5J-FRVUM7U3G/download/' +
                                            'trainingsetedgeon_nonorm_wnoise.tar.gz?pub_secret=f0820f798a')
        response = requests.get(synthetic_signals_with_noise_url)
        synthetic_signal_with_noise_compressed_data_path = Path(self.data_directory,
                                                                'TrainingSetedgeon_nonorm_wnoise.tar.gz')
        with open(synthetic_signal_with_noise_compressed_data_path, 'wb') as compressed_file:
            compressed_file.write(response.content)
        self.convert_raw_synthetic_files_to_project_format(synthetic_signal_with_noise_compressed_data_path,
                                                           'TrainingSetedgeon_nonorm_wnoise',
                                                           self.positive_data_directory)

if __name__ == '__main__':
    database = SelfLensingBinaryAllSyntheticDatabase()
    database.download_and_prepare_database_with_all_synthetic_negatives()