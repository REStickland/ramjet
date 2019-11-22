"""
Code for a database of self lensing binary synthetic data overlaid on real TESS data.
"""
import tarfile
import pandas as pd
from pathlib import Path

import tensorflow as tf

from ramjet.photometric_database.lightcurve_label_per_time_step_database import LightcurveLabelPerTimeStepDatabase


class SelfLensingBinaryDatabase(LightcurveLabelPerTimeStepDatabase):
    """
    A database of self lensing binary synthetic data overlaid on real TESS data.
    """
    def __init__(self, data_directory='data/self_lensing_binaries'):
        super().__init__(data_directory=data_directory)

    def general_preprocessing(self, example_path_tensor: tf.Tensor) -> (tf.Tensor, tf.Tensor):
        """
        Loads and preprocesses the data.

        :param example_path_tensor: The tensor containing the path to the example to load.
        :return: The example and its corresponding label.
        """
        pass

    def convert_raw_synthetic_data_to_project_format(self):
        """
        Takes the compressed synthetic data provided by Agnieszka Cieplak and converts it to a consistent form
        that's apt for the project.
        """
        path_to_agnieszka_compressed_data = Path(self.data_directory, 'TrainingSetedgeon.tar.gz')
        synthetic_data_directory = Path(self.data_directory, 'synthetic_signals')
        synthetic_data_directory.mkdir(parents=True, exist_ok=True)
        with tarfile.open(path_to_agnieszka_compressed_data, "r:gz") as tar_file:
            tar_file.extractall(path=self.data_directory)
            Path(self.data_directory, 'TrainingSetedgeon').rename(synthetic_data_directory)
        synthetic_signal_parameter_txt_path = synthetic_data_directory.joinpath('lc_parameters.txt')
        synthetic_signal_parameter_data = pd.read_csv(synthetic_signal_parameter_txt_path, delim_whitespace=True,
                                                      skipinitialspace=True,
                                                      names=['Synthetic file number', 'Lens mass (solar masses)',
                                                             'Source mass (solar masses)', 'Period (days)',
                                                             'Eccentricity', 'Inclination angle (degrees)',
                                                             'Pericenter (degrees)', 'Pericenter epoch (days)'])
        synthetic_signal_parameter_data_file_name = synthetic_signal_parameter_txt_path.with_suffix('.feather').name
        synthetic_signal_parameter_data_path = Path(self.data_directory, synthetic_signal_parameter_data_file_name)
        synthetic_signal_parameter_data.to_feather(synthetic_signal_parameter_data_path)
        synthetic_signal_parameter_txt_path.unlink()  # Delete old file.
        for synthetic_signal_txt_path in synthetic_data_directory.glob('*.out'):
            synthetic_signal = pd.read_csv(synthetic_signal_txt_path, delim_whitespace=True, skipinitialspace=True,
                                           names=['Time (hours)', 'Magnification'])
            synthetic_signal_path = synthetic_signal_txt_path.with_suffix('.feather')
            synthetic_signal.to_feather(synthetic_signal_path)
            synthetic_signal_txt_path.unlink()  # Delete old file.


if __name__ == '__main__':
    database = SelfLensingBinaryDatabase()
    database.convert_raw_synthetic_data_to_project_format()
