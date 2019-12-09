import numpy as np
import pandas as pd
import tensorflow as tf

from ramjet.photometric_database.microlensing_label_per_time_step_database import MicrolensingLabelPerTimeStepDatabase


class MicrolensingCenterLabelPerTimeStepDatabase(MicrolensingLabelPerTimeStepDatabase):

    def get_peak_centered_boolean_label(self, times: np.ndarray, lightcurve_microlensing_meta_data: pd.Series
                                        ) -> np.ndarray:
        minimum_separation_time = lightcurve_microlensing_meta_data['t0']
        closest_time_index = np.argmin(np.abs(times - minimum_separation_time))
        peak_label = np.zeros_like(times, dtype=np.bool)
        peak_label[closest_time_index] = True
        return peak_label


    def general_preprocessing(self, example_path: str) -> (tf.Tensor, tf.Tensor):
        """
        Loads and preprocesses the data.

        :param example_path_tensor: The tensor containing the path to the example to load.
        :return: The example and its corresponding label.
        """
        example_data_frame = pd.read_feather(example_path, columns=['HJD', 'flux'])
        fluxes = example_data_frame['flux'].values
        fluxes = self.normalize(fluxes)
        times = example_data_frame['HJD'].values
        time_differences = np.diff(times, prepend=times[0])
        example = np.stack([fluxes, time_differences], axis=-1)
        if self.is_positive(example_path):
            lightcurve_microlensing_meta_data = self.get_meta_data_for_lightcurve_file_path(example_path,
                                                                                            self.meta_data_frame)
            label = self.get_peak_centered_boolean_label(times, lightcurve_microlensing_meta_data)
        else:
            label = np.zeros_like(fluxes)
        return example, label

    def generate_datasets(self, positive_data_directory: str = 'positive', negative_data_directory: str = 'negative',
                          meta_data_file_path: str = 'candlist_RADec.dat.feather'
                          ) -> (tf.data.Dataset, tf.data.Dataset):
        """
        Generates the training and validation datasets.

        :param positive_data_directory: The relative path from the data directory to the directory containing the
                                        positive example files.
        :param negative_data_directory: The relative path from the data directory to the directory containing the
                                        negative example files.
        :param meta_data_file_path: The relative path from the data directory to the microlensing meta data file.
        :return: The training and validation datasets.
        """
        self.meta_data_frame = pd.read_feather(self.data_directory.joinpath(meta_data_file_path))
        positive_example_paths = list(self.data_directory.joinpath(positive_data_directory).glob('*.feather'))
        positive_example_paths = self.remove_file_paths_with_no_meta_data(positive_example_paths, self.meta_data_frame)
        print(f'{len(positive_example_paths)} positive examples.')
        negative_example_paths = list(self.data_directory.joinpath(negative_data_directory).glob('*.feather'))
        print(f'{len(negative_example_paths)} negative examples.')
        positive_datasets = self.get_training_and_validation_datasets_for_file_paths(positive_example_paths)
        positive_training_dataset, positive_validation_dataset = positive_datasets
        negative_datasets = self.get_training_and_validation_datasets_for_file_paths(negative_example_paths)
        negative_training_dataset, negative_validation_dataset = negative_datasets
        training_dataset = self.get_ratio_enforced_dataset(positive_training_dataset, negative_training_dataset,
                                                           positive_to_negative_data_ratio=1)
        validation_dataset = positive_validation_dataset.concatenate(negative_validation_dataset)
        self.spawn_processes()
        with tf.device('/CPU:0'):
            if self.trial_directory is not None:
                self.log_dataset_file_names(training_dataset, dataset_name='training')
                self.log_dataset_file_names(validation_dataset, dataset_name='validation')
            training_dataset = training_dataset.shuffle(buffer_size=len(list(training_dataset)))
            training_preprocessor = lambda file_path: tuple(tf.py_function(self.training_preprocessing,
                                                                           [file_path], [tf.float32, tf.float32]))
            training_dataset = training_dataset.map(training_preprocessor,
                                                    num_parallel_calls=16)
            training_dataset = training_dataset.padded_batch(self.batch_size, padded_shapes=([None, 2], [None])
                                                             ).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
            validation_preprocessor = lambda file_path: tuple(tf.py_function(self.evaluation_preprocessing,
                                                                             [file_path], [tf.float32, tf.float32]))
            validation_dataset = validation_dataset.map(validation_preprocessor,
                                                        num_parallel_calls=16)
            validation_dataset = validation_dataset.padded_batch(1, padded_shapes=([None, 2], [None])).prefetch(
                buffer_size=tf.data.experimental.AUTOTUNE)
        return training_dataset, validation_dataset

    def convert_single_positive_to_full_positive_label(self, peak_label):
        if np.any(peak_label):
            return np.ones_like(peak_label)
        else:
            return np.zeros_like(peak_label)

    def training_preprocessing(self, example_path_tensor: tf.Tensor) -> (tf.Tensor, tf.Tensor):
        """
        Loads and preprocesses the data for training.

        :param example_path_tensor: The tensor containing the path to the example to load.
        :return: The example and its corresponding label.
        """
        example_path = example_path_tensor.numpy().decode('utf-8')
        self.training_input_queue.put(example_path)
        example, label = self.training_output_queue.get()
        return tf.convert_to_tensor(example, dtype=tf.float32), tf.convert_to_tensor(label, dtype=tf.float32)

    def training_preprocessing_job(self, input_queue, output_queue):
        while True:
            example_path_tensor = input_queue.get()
            example, label = self.general_preprocessing(example_path_tensor)
            example, label = self.make_uniform_length_requiring_positive(
                example, label, self.time_steps_per_example, required_length_multiple_base=self.length_multiple_base
            )
            label = self.convert_single_positive_to_full_positive_label(label)
            output_queue.put((example, label))

    def evaluation_preprocessing(self, example_path_tensor: tf.Tensor) -> (tf.Tensor, tf.Tensor):
        """
        Loads and preprocesses the data for evaluation.

        :param example_path_tensor: The tensor containing the path to the example to load.
        :return: The example and its corresponding label.
        """
        example_path = example_path_tensor.numpy().decode('utf-8')
        self.validation_input_queue.put(example_path)
        example, label = self.validation_output_queue.get()
        return tf.convert_to_tensor(example, dtype=tf.float32), tf.convert_to_tensor(label, dtype=tf.float32)

    def evaluation_preprocessing_job(self, input_queue, output_queue):
        while True:
            example_path_tensor = input_queue.get()
            example, label = self.general_preprocessing(example_path_tensor)
            example, label = self.make_uniform_length_requiring_positive(
                example, label, required_length_multiple_base=self.length_multiple_base, evaluation=True
            )
            label = self.convert_single_positive_to_full_positive_label(label)
            output_queue.put((example, label))