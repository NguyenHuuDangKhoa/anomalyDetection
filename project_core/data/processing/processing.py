import pandas as pd
import numpy as np
import gc
import os
from pathlib import Path
from typing import Dict

from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.model_selection import train_test_split

import structlog
logger = structlog.getLogger(__name__)


class DataProcessor:
    """
    This class is responsible for loading data, data engineering, and feature engineering
    :attribute datasets: a variables to store different raw datasets
    :attribute full_dataset: a variable to store the full dataset, where all small datasets merged together
    """

    def __init__(self,
                 datasets: Dict = {},
                 full_dataset: pd.DataFrame = None,
                 input_vars: pd.DataFrame = None,
                 target_vars: pd.Series = None) -> None:
        self.datasets = datasets
        self.full_dataset = full_dataset
        self.X = input_vars
        self.y = target_vars
        self.uid_failed_gearbox = [918, 899, 934, 873, 862, 886, 894]
        self.uid_failed_generator = [888, 879, 899, 881, 934, 919, 938, 903, 939, 913, 892, 874, 866, 878, 945, 872, 901, 891, 870, 935, 890]

    def _load_data(self, *paths: Path) -> None:
        """
        This protected method loads all downloaded datasets and store in the dataset attribute
        :param *path: arbitrary arguments that take as many dataset locations as provided
        :return: None (raw datasets are store in datasets attribute)
        """
        for path in paths:
            self.datasets[path.split('/')[-1].split('_')[0]] = pd.read_feather(path)

    def _clean_data(self, dictionary: Dict) -> None:
        """
        This protected method cleans all datasets stored in the datasets attribute
        which include removing columns using domain knowledge,
        removing rows contain any not a number value,
        and resetting the index after cleaning
        :param dictionary: a dictionary contains names of columns to be removed
        :return: None (cleaned datasets are store in datasets attribute)
        """
        for dataset in self.datasets:
            self.datasets[dataset].drop(columns=dictionary['common']+dictionary[dataset], inplace=True)
            self.datasets[dataset].dropna(axis=0, how='any', inplace=True)
            self.datasets[dataset].reset_index(drop=True, inplace=True)

    def _merge_data(self) -> None:
        """
        This protected method merges all datasets into one dataset
        This is done using inner join and using ID and Timestamp as pivot columns
        :return: None (merged dataset is stored in full_dataset attribute)
        """
        dataset_names = list(self.datasets.keys())
        self.full_dataset = self.datasets[dataset_names[0]].copy()
        del self.datasets[dataset_names[0]]  # Free memory
        gc.collect()  # Make sure memory is flushed and make available again
        for dataset in dataset_names[1:]:
            self.full_dataset = self.full_dataset.merge(
                right=self.datasets[dataset], on=['uid', 'local_ts_start'], how='inner').copy()
            del self.datasets[dataset]  # Free memory
            gc.collect()  # Make sure memory is flushed and make available again
        self.full_dataset.set_index('local_ts_start', inplace=True)
        logger.info(f'The data now has the shape: {self.full_dataset.shape}')

    def _extract_input_target_variables(self) -> None:
        """
        This protected method extracts input and target variables
        then store them in attribute X and y respectively.
        Target variable (status) is also labelled.
        Positive cases are labelled as -1 (Minority class).
        Negative cases are labelled as 1 (Majority class).
        :return : None
        """
        # Create a new column as a target variable
        self.full_dataset['status'] = 1
        # Store input variables in X
        self.X = self.full_dataset.drop(columns=['status', 'uid']).copy()
        # Labelling target variable by comparing the actual generated power to actual wind speed
        # Any observation that has windspeed >= 5 but power_act <= 0 will be labeled as -1 (outliers/abnormal)
        self.y = self.full_dataset[['uid', 'status', 'power_act', 'windspeed_act']].copy()
        # Store target variable in y
        self.y['status'].loc[(self.y.power_act <= 0) & (self.y.windspeed_act >= 5)] = -1

    def _remove_low_variance_features(self, threshold: float = 0.0) -> None:
        """
        This protected method removes features with variance lower than specified threshold.
        By default, the threshold is set to 0.0, i.e. only constant features are removed.
        :param threshold: features with variance below this value are removed.
        :return : None
        """
        # Only remove features with variance lower than threshold
        variance_tester = VarianceThreshold(threshold)
        variance_tester.fit(self.X)
        # Keep high variance features
        non_zero_variance_features = variance_tester.get_feature_names_out()
        logger.info(f'High variance features to keep: {non_zero_variance_features}')
        logger.info(f'Total number of features to keep now: {len(non_zero_variance_features)}')
        cols_idxs = variance_tester.get_support(indices=True)
        self.X = self.X.iloc[:, cols_idxs].copy()
        logger.info(f'The data now have the shape: {self.X.shape}')

    def _remove_highly_correlated_features(self, threshold: float = 0.98) -> None:
        """
        This protected method removes features that are highly correlated to each other
        by computing pair-wise pearson correlation.
        By default, only features that are 0.98 correlated or higher will be removed.
        :param threshold: features with correlation higher than the threshold are removed.
        :return : None
        """
        # Create correlation matrix
        corr_matrix = self.X.corr().abs()
        # Select upper triangle of correlation matrix
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        # Find features with correlation greater than threshold
        to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
        # Drop features
        self.X.drop(to_drop, axis=1, inplace=True)
        logger.info(f'Highly correlated input variables to be removed: {to_drop}')
        logger.info(f'Total number of features to be removed now: {len(to_drop)}')
        logger.info(f'The data now have the shape: {self.X.shape}')

    def _supervised_filter(self, score_func: callable = f_classif, k: int = 25) -> None:
        """
        This protected method removes features by determining how well each feature
        discriminates between the classes. Depend on data types of our input and output
        different scoring functions are used. Because our data has numerical inputs and
        categorical output, we can use ANOVA or Kandall as a scoring function.
        Assuming we only want to remove input variables with low linear realationship
        to target variable, the default is set to ANOVA.
        The default number of features to keep is 25, which is determined based on
        previous experiment using the plot of feature importance.
        :score_func : scoring function.
        :k : number of best features to keep.
        :return : None
        """
        feature_selector = SelectKBest(score_func=score_func, k=k)
        feature_selector.fit(self.X, self.y.status)
        selected_features = feature_selector.get_feature_names_out()
        logger.info(f'The top 25 features to keep: {selected_features}')
        cols_idxs = feature_selector.get_support(indices=True)
        self.X = self.X.iloc[:, cols_idxs].copy()
        logger.info(f'The data now have the shape: {self.X.shape}')

    def _split_data_train_test(self,
                               output_train_data_path: Path,
                               output_test_data_path: Path,
                               use_healthy_data_only: bool = True)  -> None:
        """
        This protected method split data into testing set and training set.
        Option 1: Using only data of turbine that never failed as training set.
        Testing set consists of data of turbine that failed at least one
        Option 2: Using data of turbine as training set but utilize
        sklearn's train_test_split() function to split data
        :param option: use to choose option 1 or 2
        :return : None (training and testing sets are saved to feather file)
        """
        # All turbines that failed
        uid_failed = self.uid_failed_gearbox + self.uid_failed_generator
        # print(f'Total number of failed turbine: {len(uid_failed)}') # 899 and 934 failed both gearbox and generator
        # All turbines that not failed
        uid_not_failed = [i for i in list(self.full_dataset.uid.unique()) if i not in uid_failed]
        logger.info(f'Total number of healthy turbine: {len(uid_not_failed)}')

        if use_healthy_data_only:
            # Create training set
            self.X['uid'] = self.full_dataset['uid']
            X_all_class_1 = self.X.loc[self.X['uid'].isin(uid_not_failed)].copy()
            X_all_class_1.drop(columns=['uid'], inplace=True)

            y_all_class_1 = self.y.loc[self.y['uid'].isin(uid_not_failed)]
            self.y = self.y['status']
            y_all_class_1 = y_all_class_1['status']
            X_train = X_all_class_1.reset_index()
            y_train = y_all_class_1.reset_index()
            logger.info(f'X has shape: {self.X.shape}')
            logger.info(f'X_train has shape: {X_all_class_1.shape}')
            logger.info(f'y has shape: {self.y.shape}')
            logger.info(f'y_train has shape: {y_all_class_1.shape}')

            # Create testing set
            X_test = self.X.loc[self.X['uid'].isin(set(uid_failed))].copy()
            self.X.drop(columns=['uid'], inplace=True)
            X_test.drop(columns=['uid'], inplace=True)

            y_temporary = self.y.to_frame()
            y_temporary['uid'] = self.full_dataset['uid']
            y_test = y_temporary.loc[y_temporary['uid'].isin(set(uid_failed))].copy()
            y_test.drop(columns=['uid'], inplace=True)
            del y_temporary
            X_test.reset_index(inplace=True)
            y_test.reset_index(inplace=True)
            logger.info(f'X_test has shape: {X_test.shape}')
            logger.info(f'y_test has shape: {y_test.shape}')

            # Save X_train, X_test, y_train, y_test as feather files
            X_train.to_feather(path=os.path.join(output_train_data_path, 'X_train.feather'))
            y_train.to_feather(path=os.path.join(output_train_data_path, 'y_train.feather'))
            X_test.to_feather(path=os.path.join(output_test_data_path, 'X_test.feather'))
            y_test.to_feather(path=os.path.join(output_test_data_path, 'y_test.feather'))

        else:
            # Split into train, validation, and test sets
            # Split ratio 7/3 to ensure the validation and test sets have enough both healthy and faulty data points
            X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, train_size=0.7, shuffle=True, stratify=self.y)
            X_train.to_feather(path=os.path.join(output_train_data_path, 'X_train.feather'))
            y_train.to_feather(path=os.path.join(output_train_data_path, 'y_train.feather'))
            X_test.to_feather(path=os.path.join(output_test_data_path, 'X_test.feather'))
            y_test.to_feather(path=os.path.join(output_test_data_path, 'y_test.feather'))
