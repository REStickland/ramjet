"""
Code for a database of self lensing binary synthetic data overlaid on real TESS data.
"""
import os
import tarfile
from typing import Union, List

import numpy as np
import pandas as pd
import requests
import tensorflow as tf
from pathlib import Path

from astroquery.mast import Catalogs
from scipy.interpolate import interp1d

from ramjet.photometric_database.tess_lightcurve_label_per_time_step_database import \
    TessLightcurveLabelPerTimeStepDatabase


class SelfLensingBinaryDatabase(TessLightcurveLabelPerTimeStepDatabase):
    """
    A database of self lensing binary synthetic data overlaid on real TESS data.
    """

    def __init__(self, data_directory='data/self_lensing_binaries'):
        super().__init__(data_directory=data_directory)
        self.synthetic_signals_directory: Path = Path(self.data_directory, 'synthetic_signals')
        self.training_synthetic_signal_paths = np.array([])

    def generate_datasets(self) -> (tf.data.Dataset, tf.data.Dataset):
        """
        Generates the training and validation datasets.

        :return: The training and validation datasets.
        """
        all_lightcurve_paths = list(map(str, self.lightcurve_directory.glob('*.fits')))
        all_signal_paths = list(map(str, self.synthetic_signals_directory.glob('*.feather')))
        training_lightcurve_paths, validation_lightcurve_paths = self.extract_chunk_and_remainder(all_lightcurve_paths,
                                                                                                  self.validation_ratio)
        training_signal_paths, validation_signal_paths = self.extract_chunk_and_remainder(all_signal_paths,
                                                                                          self.validation_ratio)
        print(f'{len(training_lightcurve_paths)} training lightcurves.')
        print(f'{len(validation_lightcurve_paths)} validation lightcurves.')
        print(f'{len(training_signal_paths)} training synthetic signals.')
        print(f'{len(validation_signal_paths)} validation synthetic signals.')
        ordered_validation_signal_paths = np.random.choice(validation_signal_paths,
                                                           size=validation_lightcurve_paths.shape[0])
        validation_combined_paths = np.stack([validation_lightcurve_paths, ordered_validation_signal_paths], axis=1)
        self.training_synthetic_signal_paths = training_signal_paths
        training_dataset = tf.data.Dataset.from_tensor_slices(training_lightcurve_paths)
        validation_dataset = tf.data.Dataset.from_tensor_slices(validation_combined_paths)

        if self.trial_directory is not None:
            self.log_dataset_file_names(training_dataset, dataset_name='training_lightcurves')
            self.log_dataset_file_names(training_signal_paths, dataset_name='training_synthetic_signals')
            self.log_dataset_pair_file_names(validation_dataset, dataset_name='validation')
        training_dataset = training_dataset.shuffle(buffer_size=len(list(training_dataset)))
        training_preprocessor = lambda file_path: tf.py_function(self.dual_training_preprocessing,
                                                                 [file_path], [tf.float32])
        training_dataset = training_dataset.map(training_preprocessor, num_parallel_calls=16)
        training_dataset = training_dataset.flat_map(lambda x: tf.data.Dataset.from_tensor_slices(x))
        training_unstack_preprocessor = lambda example_and_label_tensor: tuple(
            tf.py_function(self.unstack_example_and_label,
                           [example_and_label_tensor], [tf.float32, tf.float32]))
        training_dataset = training_dataset.map(training_unstack_preprocessor, num_parallel_calls=16)
        training_dataset = training_dataset.padded_batch(self.batch_size, padded_shapes=([None, 2], [None])).prefetch(
            buffer_size=tf.data.experimental.AUTOTUNE)
        validation_preprocessor = lambda file_path: tf.py_function(self.dual_evaluation_preprocessing,
                                                                   [file_path], [tf.float32])
        validation_dataset = validation_dataset.map(validation_preprocessor, num_parallel_calls=4)
        validation_dataset = validation_dataset.flat_map(lambda x: tf.data.Dataset.from_tensor_slices(x))
        validation_unstack_preprocessor = lambda example_and_label_tensor: tuple(
            tf.py_function(self.unstack_example_and_label,
                           [example_and_label_tensor], [tf.float32, tf.float32]))
        validation_dataset = validation_dataset.map(validation_unstack_preprocessor, num_parallel_calls=16)
        validation_dataset = validation_dataset.padded_batch(1, padded_shapes=([None, 2], [None])).prefetch(
            buffer_size=tf.data.experimental.AUTOTUNE)
        return training_dataset, validation_dataset

    def log_dataset_pair_file_names(self, dataset: tf.data.Dataset, dataset_name: str):
        os.makedirs(self.trial_directory, exist_ok=True)
        lightcurve_paths = [lightcurve_path.numpy().decode('utf-8') for lightcurve_path, _ in list(dataset)]
        synthetic_signal_paths = [synthetic_signal_path.numpy().decode('utf-8') for _, synthetic_signal_path
                                  in list(dataset)]
        lightcurve_series = pd.Series(lightcurve_paths)
        lightcurve_series.to_csv(os.path.join(self.trial_directory, f'{dataset_name}_lightcurves.csv'),
                                 header=False, index=False)
        synthetic_signal_series = pd.Series(synthetic_signal_paths)
        synthetic_signal_series.to_csv(os.path.join(self.trial_directory, f'{dataset_name}_synthetic_signals.csv'),
                                       header=False, index=False)

    def unstack_example_and_label(self, example_and_label_tensor: tf.Tensor) -> (np.ndarray, np.ndarray):
        example_and_label = example_and_label_tensor.numpy()
        example = example_and_label[:, :2]
        label = example_and_label[:, 2]
        example_tensor = tf.convert_to_tensor(example, dtype=tf.float32)
        label_tensor = tf.convert_to_tensor(label, dtype=tf.float32)
        return example_tensor, label_tensor

    def dual_general_preprocessing(self, lightcurve_path: str, synthetic_signal_path: str) -> (np.ndarray, np.ndarray,
                                                                                               np.ndarray, np.ndarray):
        fluxes, times = self.load_fluxes_and_times_from_fits_file(lightcurve_path)
        median_flux = np.median(fluxes)
        signal_dataframe = pd.read_feather(synthetic_signal_path)
        signal_magnifications = signal_dataframe['Magnification'].values
        signal_times = signal_dataframe['Time (hours)'].values
        signal_times /= 24  # Convert from hours to days.
        signal_fluxes = (signal_magnifications - 1) * median_flux
        time_differences = np.diff(times, prepend=times[0])
        signal_flux_interpolator = interp1d(signal_times, signal_fluxes, bounds_error=True)
        interpolated_signal_fluxes = signal_flux_interpolator(time_differences)
        fluxes_with_injected_signal = fluxes + interpolated_signal_fluxes
        fluxes = self.normalize(fluxes)
        fluxes_with_injected_signal = self.normalize(fluxes_with_injected_signal)
        example = np.stack([fluxes, time_differences], axis=-1)
        example_with_injected_signal = np.stack([fluxes_with_injected_signal, time_differences], axis=-1)
        return example, np.zeros_like(fluxes), example_with_injected_signal, np.ones_like(fluxes)

    def dual_training_preprocessing(self, lightcurve_path_tensor: tf.Tensor) -> (
            (tf.Tensor, tf.Tensor), (tf.Tensor, tf.Tensor)):
        synthetic_signal_to_inject_path = np.random.choice(self.training_synthetic_signal_paths)
        lightcurve_path = lightcurve_path_tensor.numpy().decode('utf-8')
        examples_and_labels = self.dual_general_preprocessing(lightcurve_path, synthetic_signal_to_inject_path)
        negative_example, negative_label, positive_example, positive_label = examples_and_labels
        negative_example, negative_label, positive_example, positive_label = self.make_uniform_length_in_uniform(
            negative_example, negative_label, positive_example, positive_label, length=self.time_steps_per_example)
        negative_example_and_label = np.concatenate([negative_example, np.expand_dims(negative_label, axis=-1)], axis=1)
        positive_example_and_label = np.concatenate([positive_example, np.expand_dims(positive_label, axis=-1)], axis=1)
        joint_examples_and_labels = np.stack([negative_example_and_label, positive_example_and_label], axis=0)
        return joint_examples_and_labels

    def dual_evaluation_preprocessing(self, lightcurve_path_tensor: tf.Tensor) -> (tf.Tensor, tf.Tensor):
        lightcurve_path = lightcurve_path_tensor[0].numpy().decode('utf-8')
        synthetic_signal_path = lightcurve_path_tensor[1].numpy().decode('utf-8')
        examples_and_labels = self.dual_general_preprocessing(lightcurve_path, synthetic_signal_path)
        negative_example, negative_label, positive_example, positive_label = examples_and_labels
        negative_example, negative_label, positive_example, positive_label = self.make_uniform_length_in_uniform(
            negative_example, negative_label, positive_example, positive_label, evaluation=True)
        negative_example_and_label = np.concatenate([negative_example, np.expand_dims(negative_label, axis=-1)], axis=1)
        positive_example_and_label = np.concatenate([positive_example, np.expand_dims(positive_label, axis=-1)], axis=1)
        joint_examples_and_labels = np.stack([negative_example_and_label, positive_example_and_label], axis=0)
        return joint_examples_and_labels

    def make_uniform_length_in_uniform(self, negative_example, negative_label, positive_example, positive_label,
                                       length: Union[int, None] = None, evaluation: bool = False):
        negative_and_positive_arrays = np.concatenate([negative_example, np.expand_dims(negative_label, axis=-1),
                                                       positive_example, np.expand_dims(positive_label, axis=-1)],
                                                      axis=1)
        negative_and_positive_arrays = self.make_uniform_length_with_multiple(
            negative_and_positive_arrays, length,
            required_length_multiple_base=self.length_multiple_base,
            evaluation=evaluation
        )
        negative_example, negative_label = negative_and_positive_arrays[:, :2], negative_and_positive_arrays[:, 2]
        positive_example, positive_label = negative_and_positive_arrays[:, 3:5], negative_and_positive_arrays[:, 5]
        return negative_example, negative_label, positive_example, positive_label

    def make_uniform_length_with_multiple(self, array: np.ndarray,
                                          length: Union[int, None] = None,
                                          required_length_multiple_base: Union[int, None] = None,
                                          evaluation: bool = False) -> (np.ndarray, np.ndarray):
        if length is None:
            length = array.shape[0]
        if required_length_multiple_base is not None:
            length = self.round_to_base(length, base=required_length_multiple_base)
        if length == array.shape[0]:
            return array
        array = self.make_uniform_length(array, length, randomize=not evaluation)
        return array

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

    def download_and_prepare_database(self, number_of_negative_lightcurves_to_download=10000,
                                      magnitude_filter: (float, float) = None):
        print('Clearing data directory...')
        self.clear_data_directory()
        print('Downloading synthetic signals...')
        synthetic_signals_url = ('https://files.slack.com/files-pri/TKNTDRH5J-FQR2KN5BJ/download/' +
                                 'trainingsetedgeon.tar.gz?pub_secret=81aa782027')
        response = requests.get(synthetic_signals_url)
        path_to_agnieszka_compressed_data = Path(self.data_directory, 'TrainingSetedgeon.tar.gz')
        with open(path_to_agnieszka_compressed_data, 'wb') as compressed_file:
            compressed_file.write(response.content)
        self.convert_raw_synthetic_data_to_project_format()
        print('Downloading TESS observation list...')
        tess_observations = self.get_all_tess_time_series_observations()
        if magnitude_filter is not None:
            timeseries_tic_entries = self.get_all_tic_entries_by_tic_ids(
                tic_ids=tess_observations['target_name'].values)
            magnitude_selected_tic_entries = timeseries_tic_entries[
                timeseries_tic_entries['Tmag'].between(*magnitude_filter)]
            tess_observations = self.get_all_tess_time_series_observations(
                {'target_name': magnitude_selected_tic_entries['ID'].values})
        single_sector_observations = self.get_single_sector_observations(tess_observations)
        single_sector_observations = self.add_sector_column_based_on_single_sector_obs_id(single_sector_observations)
        single_sector_observations['tic_id'] = single_sector_observations['target_name'].astype(int)
        single_sector_observations = single_sector_observations.sample(frac=1, random_state=0)  # Shuffle
        # Shorten product list obtaining. Twice as much *should* be plenty to find what we need.
        single_sector_observations = single_sector_observations.head(number_of_negative_lightcurves_to_download * 2)
        single_sector_data_products = self.get_product_list(single_sector_observations)
        single_sector_data_products = self.add_tic_id_column_based_on_single_sector_obs_id(single_sector_data_products)
        single_sector_data_products = single_sector_data_products[
            single_sector_data_products['productFilename'].str.endswith('lc.fits')
        ]
        single_sector_data_products = single_sector_data_products.sample(frac=1, random_state=0)  # Shuffle
        download_manifest = self.download_products(
            single_sector_data_products.head(number_of_negative_lightcurves_to_download)
        )
        print(f'Moving lightcurves to {self.lightcurve_directory}...')
        for file_path_string in download_manifest['Local Path']:
            file_path = Path(file_path_string)
            file_path.rename(self.lightcurve_directory.joinpath(file_path.name))
        print('Database ready.')

    def general_preprocessing(self, example_path_tensor: tf.Tensor) -> (tf.Tensor, tf.Tensor):
        raise NotImplementedError


if __name__ == '__main__':
    database = SelfLensingBinaryDatabase()
    database.download_and_prepare_database()
