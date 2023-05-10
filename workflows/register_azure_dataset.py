"""
The get_data function should load the raw data in any format from any place and return it
"""
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import List, Text

import pandas as pd
import structlog
from azureml.data import TabularDataset

from project_core.cloud.azure.utils import (get_aml_datastore,
                                            get_aml_workspace,
                                            register_aml_dataset)
from project_core.utils import load_yaml_config

logger = structlog.getLogger(__name__)

#FIXME
def parse_arguments(arguments_list: List) -> Namespace:
    """Parse arguments list"""
    parser = ArgumentParser(__name__)
    parser.add_argument('--config_path', type=Path, help='Path to run_config file')

    return parser.parse_args(arguments_list)

#FIXME
def register_dataset(
    datastore_name: Text,
    file_path: Text,
    dataset_name: Text,
) -> TabularDataset:
    """
    Register dataset to AzureML Datasets from the configured datastore path
    :param datastore_name: Name of the datastore where the data currently live
    :param file_path: Path of the data file within the datastore
    :param dataset_name: Name of the dataset to register data to
    :return: Registered dataset object
    """
    aml_ws = get_aml_workspace()

    aml_datastore = get_aml_datastore(
        workspace=aml_ws,
        name=datastore_name)

    registered_dataset = register_aml_dataset(
        workspace=aml_ws,
        datastore=aml_datastore,
        file_path=file_path,
        dataset_name=dataset_name,
    )

    return registered_dataset

#FIXME
def run(arguments_list: List = None):
    """Run step"""
    arguments = parse_arguments(arguments_list=arguments_list)
    config = load_yaml_config(config_path=arguments.config_path)
    registered_dataset = register_dataset(
        datastore_name=config.get('default_azure_configs').get('datastore_name'),
        file_path=config.get('default_azure_configs').get('file_path'),
        dataset_name=config.get('default_azure_configs').get('dataset_name'),
    )
    logger.info(registered_dataset)


if __name__ == '__main__':
    run()
