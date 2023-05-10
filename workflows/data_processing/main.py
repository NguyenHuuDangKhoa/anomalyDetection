"""
This file should implement the data processing pipeline. So given raw data in any format,
perform whatever processing is needed in split_data and return the processed data in any format.
"""

from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import List
import structlog
import os
from project_core.data.processing import processing
from project_core.data.processing import processing_nbm_rul
from project_core.utils import load_yaml_config
from sklearn.feature_selection import f_classif

logger = structlog.getLogger(__name__)


def parse_arguments(arguments_list: List) -> Namespace:
    """
    This function parses the arguments list created in local_pipeline.py,
    which get the arguments from the pipeline config yaml file.
    :param arguments_list: list of arguments
    :return: a Namespace that defines all required argurments for data processing core file
    """
    parser = ArgumentParser(__name__)
    parser.add_argument('--output_train_data_path', type=Path, help='output train data path')
    parser.add_argument('--output_test_data_path', type=Path, help='output test data path')
    parser.add_argument('--config_path', type=str, help='Path to data processing config file')
    parser.add_argument('--approach', type=str,
                        choices=['one_class_classification', 'NBM_RUL'],
                        help='General direction to process and train data')
    return parser.parse_args(arguments_list)


def main(arguments_list: List = None):
    """Run step"""
    logger.info('In data_processing.main.py')
    arguments = parse_arguments(arguments_list=arguments_list)
    config = load_yaml_config(config_path=Path(os.path.join(Path(__file__).parent, arguments.config_path)))
    if arguments.approach == 'one_class_classification':
        logger.info('Choosing One Class Classification Direction')
        # Packing the arguments to a List
        input_data_paths = [*config.get('data_processing_settings').get('input_data_paths').values()]
        drop_columns = config.get('data_processing_settings').get('drop_columns')
        logger.info('Run processing core')
        logger.info('Create processor instance')
        data_processor = processing.DataProcessor()
        logger.info('Load all datasets')
        data_processor._load_data(*input_data_paths)
        logger.info('Clean all datasets')
        data_processor._clean_data(dictionary=drop_columns)
        logger.info('Merge all datasets')
        data_processor._merge_data()
        logger.info('Extract Input and Target Variables')
        data_processor._extract_input_target_variables()
        logger.info('Remove low variance features')
        data_processor._remove_low_variance_features(threshold=0.0)
        logger.info('Remove highly correlated features')
        data_processor._remove_highly_correlated_features(threshold=0.98)
        logger.info('Select features using filter method of supervised feature selection')
        data_processor._supervised_filter(score_func=f_classif, k=25)
        logger.info('Split data into training and testing sets')
        data_processor._split_data_train_test(
            output_train_data_path=arguments.output_train_data_path,
            output_test_data_path=arguments.output_test_data_path,
            use_healthy_data_only=True)
        logger.info('Save training and testing sets to feather files')
    elif arguments.approach == 'NBM_RUL':
        logger.info('Choosing Normal Behaviour Model - Remaining Useful Life Direction')
        input_data_paths = [*config.get('data_processing_settings').get('input_data_paths').values()]
        excel_failure_data_paths = [*config.get('data_processing_settings').get('excel_failure_data_paths').values()]
        csv_failure_data_paths = [*config.get('data_processing_settings').get('csv_failure_data_paths').values()]
        drop_columns = config.get('data_processing_settings').get('drop_columns')
        failure_settings = config.get('data_processing_settings').get('failure_settings')
        nbm_settings = config.get('data_processing_settings').get('nbm_settings')
        rul_column = config.get('data_processing_settings').get('rul_column')
        data_split = config.get('data_processing_settings').get('data_split')
        logger.info('Run processing core')
        logger.info('Create processor instance')
        data_processor = processing_nbm_rul.DataProcessorNBM()
        logger.info('Load all datasets')
        data_processor._load_data(*input_data_paths)
        logger.info('Clean all datasets')
        data_processor._clean_data(dictionary=drop_columns)
        logger.info('Merge all datasets')
        data_processor._merge_data()
        logger.info('Load in excel datasets')
        data_processor._load_failure_excel(*excel_failure_data_paths)
        logger.info('Transforming excel datasets')
        data_processor._transform_failure_excel()
        logger.info('Load in csv datasets')
        data_processor._load_failure_csv(*csv_failure_data_paths)
        logger.info('Combining failure data')
        data_processor._combine_failure_data(dictionary=failure_settings)
        logger.info('Remove highly correlated features')
        data_processor._remove_highly_correlated_features(threshold=0.95)
        logger.info('Create unhealthy dataset')
        data_processor._create_unhealthy_data()
        logger.info('Create NBM dataset')
        data_processor._create_nbm_data(dictionary=nbm_settings)
        logger.info('Split data into training and testing sets')
        data_processor._split_data_train_test(
            output_train_data_path=arguments.output_train_data_path,
            output_test_data_path=arguments.output_test_data_path,
            model_type='nbm',
            dictionary=data_split)
        logger.info('Save training and testing sets to feather files')

if __name__ == '__main__':
    main()
