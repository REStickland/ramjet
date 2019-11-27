"""
Code for a database of self lensing binary synthetic data overlaid on real TESS data.
"""
import tarfile
import pandas as pd
import requests
import tensorflow as tf
from pathlib import Path

from ramjet.photometric_database.tess_lightcurve_label_per_time_step_database import \
    TessLightcurveLabelPerTimeStepDatabase


class SelfLensingBinaryDatabase(TessLightcurveLabelPerTimeStepDatabase):
    """
    A database of self lensing binary synthetic data overlaid on real TESS data.
    """

    def __init__(self, data_directory='data/self_lensing_binaries'):
        super().__init__(data_directory=data_directory)
        self.synthetic_signals_directory: Path = Path(self.data_directory, 'synthetic_signals')

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

    def download_and_prepare_database(self, number_of_negative_lightcurves_to_download=10000):
        print('Clearing data directory...')
        self.clear_data_directory()
        print('Downloading synthetic signals.')
        synthetic_signals_url = ('https://files.slack.com/files-pri/TKNTDRH5J-FQR2KN5BJ/download/' +
                                 'trainingsetedgeon.tar.gz?pub_secret=81aa782027')
        response = requests.get(synthetic_signals_url)
        path_to_agnieszka_compressed_data = Path(self.data_directory, 'TrainingSetedgeon.tar.gz')
        with open(path_to_agnieszka_compressed_data, 'wb') as compressed_file:
            compressed_file.write(response.content)
        self.convert_raw_synthetic_data_to_project_format()
        print('Downloading TESS observation list...')
        tess_observations = self.get_all_tess_time_series_observations()
        single_sector_observations = self.get_single_sector_observations(tess_observations)
        single_sector_observations = self.add_sector_column_based_on_single_sector_obs_id(single_sector_observations)
        single_sector_observations['tic_id'] = single_sector_observations['target_name'].astype(int)
        single_sector_observations = single_sector_observations.sample(frac=1, random_state=0)  # Shuffle
        # Shorten product list obtaining. Twice as much *should* be plenty to find what we need.
        single_sector_observations = single_sector_observations.head(number_of_negative_lightcurves_to_download * 2)
        single_sector_data_products = self.get_product_list(single_sector_observations)
        single_sector_data_products = self.add_tic_id_column_based_on_single_sector_obs_id(single_sector_data_products)
        single_sector_data_products = single_sector_data_products.sample(frac=1, random_state=0)  # Shuffle
        download_manifest = self.download_products(
            single_sector_data_products.head(number_of_negative_lightcurves_to_download)
        )
        print(f'Moving lightcurves to {self.lightcurve_directory}...')
        for file_path_string in download_manifest['Local Path']:
            file_path = Path(file_path_string)
            file_path.rename(self.lightcurve_directory.joinpath(file_path.name))
        print('Database ready.')


if __name__ == '__main__':
    database = SelfLensingBinaryDatabase()
    database.download_and_prepare_database()
