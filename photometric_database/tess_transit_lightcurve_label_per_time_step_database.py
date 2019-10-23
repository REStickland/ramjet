"""
Code for a database of TESS transit lightcurves with a label per time step.
"""
from pathlib import Path
from typing import List
import pandas as pd
import requests
from astropy.table import Table
from astroquery.mast import Observations
from astroquery.exceptions import TimeoutError as AstroQueryTimeoutError
from requests.exceptions import ConnectionError

from photometric_database.lightcurve_label_per_time_step_database import LightcurveLabelPerTimeStepDatabase


class TessTransitLightcurveLabelPerTimeStepDatabase(LightcurveLabelPerTimeStepDatabase):
    """
    A class for a database of TESS transit lightcurves with a label per time step.
    """

    def __init__(self, data_directory='data/tess'):
        super().__init__()
        self.data_directory = Path(data_directory)
        self.data_directory.mkdir(parents=True, exist_ok=True)
        self.lightcurve_directory = self.data_directory.joinpath('lightcurves')
        self.lightcurve_directory.mkdir(parents=True, exist_ok=True)
        self.data_validation_directory = self.data_directory.joinpath('data_validations')
        self.data_validation_directory.mkdir(parents=True, exist_ok=True)
        self.data_validation_dictionary = None

    def get_lightcurve_file_paths(self) -> List[Path]:
        """
        Gets all the file paths for the available lightcurves.
        """
        return list(self.lightcurve_directory.glob('*.fits'))

    def obtain_data_validation_dictionary(self):
        """
        Collects all the data validation files into a dictionary for fast TIC ID lookup.
        """
        data_validation_dictionary = {}
        for path in self.data_validation_directory.glob('*.xml'):
            tic_id = path.name.split('-')[3]  # The TIC ID is just in the middle of the file name.
            data_validation_dictionary[tic_id] = path
        self.data_validation_dictionary = data_validation_dictionary

    def is_positive(self, example_path: str) -> bool:
        """
        Checks if an example contains a transit event or not.

        :param example_path: The path to the example to check.
        :return: Whether or not the example contains a transit event.
        """
        tic_id = str(Path(example_path).name).split('-')[2]  # The TIC ID is just in the middle of the file name.
        return tic_id in self.data_validation_dictionary

    def download_liang_yu_database(self):
        """
        Downloads the database used by https://arxiv.org/pdf/1904.02726.pdf.
        """
        print("Downloading Liang Yu's disposition CSV...")
        liang_yu_csv_url = 'https://raw.githubusercontent.com/yuliang419/Astronet-Triage/master/astronet/tces.csv'
        response = requests.get(liang_yu_csv_url)
        liang_yu_disposition_path = self.data_directory.joinpath('liang_yu_disposition.csv')
        with open(liang_yu_disposition_path, 'wb') as csv_file:
            csv_file.write(response.content)
        print('Downloading TESS observation list...')
        # tess_observations = self.get_all_tess_time_series_observations()
        tess_observations = pd.read_feather('/Users/golmschenk/Desktop/tess_time_series_observations.feather')
        single_sector_observations = self.get_single_sector_observations(tess_observations)
        single_sector_observations = self.add_sector_column_based_on_single_sector_obs_id(single_sector_observations)
        single_sector_observations['tic_id'] = single_sector_observations['target_name'].astype(int)
        print("Downloading lightcurves which appear in Liang Yu's disposition...")
        # noinspection SpellCheckingInspection
        columns_to_use = ['tic_id', 'Disposition', 'Epoc', 'Period', 'Duration', 'Sectors']
        liang_yu_disposition = pd.read_csv(liang_yu_disposition_path, usecols=columns_to_use)
        liang_yu_observations = pd.merge(single_sector_observations, liang_yu_disposition, how='inner',
                                         left_on=['tic_id', 'sector'], right_on=['tic_id', 'Sectors'])
        number_of_observations_not_found = liang_yu_disposition.shape[0] - liang_yu_observations.shape[0]
        print(f"{liang_yu_observations.shape[0]} observations found that match Liang Yu's entries.")
        print(f'Liang Yu used the FFIs, not the lightcurve products, so many will be missing.')
        print(f"No observations found for {number_of_observations_not_found} entries in Liang Yu's disposition.")
        liang_yu_data_products = self.get_data_products(liang_yu_observations)
        liang_yu_lightcurve_data_products = liang_yu_data_products[
            liang_yu_data_products['productFilename'].str.endswith('lc.fits')
        ]
        download_manifest = self.download_products(liang_yu_lightcurve_data_products)
        print('Moving lightcurves to {self.lightcurve_directory}...')
        for file_path_string in download_manifest['Local Path']:
            file_path = Path(file_path_string)
            file_path.rename(self.lightcurve_directory.joinpath(file_path.name))
        print('Database ready.')

    @staticmethod
    def get_all_tess_time_series_observations() -> pd.DataFrame:
        """
        Gets all TESS time-series observations, limited to science data product level. Repeats download attempt on
        error.
        """
        tess_observations = None
        while tess_observations is None:
            try:
                # noinspection SpellCheckingInspection
                tess_observations = Observations.query_criteria(obs_collection='TESS', dataproduct_type='timeseries',
                                                                calib_level=3)  # Science data product level.
            except (AstroQueryTimeoutError, ConnectionError):
                print('Error connecting to MAST. They have occasional downtime. Trying again...')
        return tess_observations.to_pandas()

    @staticmethod
    def get_data_products(observations: pd.DataFrame) -> pd.DataFrame:
        """
        A wrapper for MAST's `get_data_products`, allowing the use of Pandas DataFrames instead of AstroPy Tables.
        Retries on error when communicating with the MAST server.

        :param observations: The data frame of observations to get. Will be converted from DataFrame to Table for query.
        :return: The data frame of the product list. Will be converted from Table to DataFrame for use.
        """
        data_products = None
        while data_products is None:
            try:
                # noinspection SpellCheckingInspection
                data_products = Observations.get_product_list(Table.from_pandas(observations))
            except (AstroQueryTimeoutError, ConnectionError):
                print('Error connecting to MAST. They have occasional downtime. Trying again...')
        return data_products.to_pandas()

    def download_products(self, data_products: pd.DataFrame) -> pd.DataFrame:
        """
         A wrapper for MAST's `download_products`, allowing the use of Pandas DataFrames instead of AstroPy Tables.
        Retries on error when communicating with the MAST server.

        :param data_products: The data frame of data products to download. Will be converted from DataFrame to Table
                              for sending the request to MAST.
        :return: The manifest of the download. Will be converted from Table to DataFrame for use.
        """
        manifest = None
        while manifest is None:
            try:
                # noinspection SpellCheckingInspection
                manifest = Observations.download_products(Table.from_pandas(data_products),
                                                          download_dir=str(self.data_directory))
            except (AstroQueryTimeoutError, ConnectionError):
                print('Error connecting to MAST. They have occasional downtime. Trying again...')
        return manifest.to_pandas()

    @staticmethod
    def get_single_sector_observations(time_series_observations: pd.DataFrame) -> pd.DataFrame:
        """
        Filters a data frame of observations to get only the single sector observations.

        :param time_series_observations: A data frame of observations to filter for single sector observations.
        :return: The data frame of single sector observations.
        """
        single_sector_observations = time_series_observations[
            time_series_observations['dataURL'].str.endswith('lc.fits')
        ]
        return single_sector_observations.copy()

    @staticmethod
    def get_multi_sector_observations(time_series_observations: pd.DataFrame) -> pd.DataFrame:
        """
        Filters a data frame of observations to get only the multi sector observations.

        :param time_series_observations: A data frame of observations to filter for multi sector observations.
        :return: The data frame of multi sector observations.
        """
        multi_sector_observations = time_series_observations[
            time_series_observations['dataURL'].str.endswith('dvt.fits')
        ]
        return multi_sector_observations.copy()

    def add_sector_column_based_on_single_sector_obs_id(self, observations: pd.DataFrame) -> pd.DataFrame:
        """
        Adds a column with the sector the data was taken from.

        :param observations: The table of single-sector observations.
        :return: The table with the added sector column.
        """
        observations['sector'] = observations['obs_id'].map(self.get_sector_from_single_sector_obs_id)
        return observations

    def add_tic_id_column_based_on_single_sector_obs_id(self, data_frame: pd.DataFrame) -> pd.DataFrame:
        """
        Adds a (string) column with the TIC ID the row is related to.

        :param data_frame: The data frame of single-sector entries.
        :return: The table with the added TIC ID column.
        """
        data_frame['tic_id'] = data_frame['obs_id'].map(self.get_tic_id_from_single_sector_obs_id)
        return data_frame

    @staticmethod
    def get_tic_id_from_single_sector_obs_id(obs_id: str) -> str:
        """
        Extracts the TIC ID from a single-sector obs_id string.

        :param obs_id: The obs_id to extract from.
        :return: The extracted TIC ID.
        """
        return obs_id.split('-')[2].lstrip('0')

    def add_sector_columns_based_on_multi_sector_obs_id(self, observations: pd.DataFrame) -> pd.DataFrame:
        """
        Adds columns with sector information the data was taken from. In particular, adds the start and end
        sectors, as well as the total length of the sector range.

        :param observations: The data frame of multi-sector observations.
        :return: The data frame with the added sector information columns.
        """
        sectors_data_frame = observations['obs_id'].apply(self.get_sectors_from_multi_sector_obs_id)
        observations['start_sector'] = sectors_data_frame[0]
        observations['end_sector'] = sectors_data_frame[1]
        observations['sector_range_length'] = observations['end_sector'] - observations['start_sector'] + 1
        return observations

    @staticmethod
    def get_sector_from_single_sector_obs_id(obs_id: str) -> int:
        """
        Extracts the sector from a single-sector obs_id string.

        :param obs_id: The obs_id to extract from.
        :return: The extracted sector number.
        """
        return int(obs_id.split('-')[1][1:])

    @staticmethod
    def get_sectors_from_multi_sector_obs_id(obs_id: str) -> pd.Series:
        """
        Extracts the sectors from a multi-sector obs_id string.

        :param obs_id: The obs_id to extract from.
        :return: The extracted sector numbers: a start and an end sector.
        """
        string_split = obs_id.split('-')
        return pd.Series([int(string_split[1][1:]), int(string_split[2][1:])])

    def get_largest_sector_range(self, multi_sector_observations: pd.DataFrame) -> pd.DataFrame:
        """
        Returns only the rows with the largest sector range for each TIC ID.

        :param multi_sector_observations: The observations with sector range information included.
        :return: A data frame containing only the rows for each TIC ID that have the largest sector range.
        """
        range_sorted_observations = multi_sector_observations.sort_values('sector_range_length', ascending=False)
        return range_sorted_observations.drop_duplicates(['target_name'])


if __name__ == '__main__':
    TessTransitLightcurveLabelPerTimeStepDatabase().download_liang_yu_database()
