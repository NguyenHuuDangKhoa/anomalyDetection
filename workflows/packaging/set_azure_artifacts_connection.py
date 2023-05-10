"""Setup Azure Artifacts connection"""

import os
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import List

import structlog
from azureml.core import Workspace
from dotenv import load_dotenv

from project_core.utils import load_yaml_config

logger = structlog.get_logger(__name__)

#FIXME
def parse_arguments(arguments_list: List) -> Namespace:
    """Parse arguments list"""
    parser = ArgumentParser(__name__)
    parser.add_argument('--config_path', type=Path, help='Path to data processing config file')

    return parser.parse_args(arguments_list)

#FIXME
def set_azure_artifacts_connection():
    """
    This method sets the connection to the Azure artifacts
    """
    arguments = parse_arguments(arguments_list=None)
    config = load_yaml_config(config_path=Path(__file__).parent / arguments.config_path)

    # Create workspace object
    ws = Workspace.from_config()

    # Load the environment variables
    load_dotenv()

    # Add azure artifacts connection to the workspace
    ws.set_connection(
        name=config.get('AZURE_ARTIFACTS_CONNECTION_NAME'),
        category='PythonFeed',
        target=config.get('AZURE_ARTIFACTS_TARGET_CONNECTION'),
        authType='PAT',
        value=os.environ.get('PAT_TOKEN'),
    )

    logger.info('Added azure artifacts connection to the workspace',
                connection_name=config.get('AZURE_ARTIFACTS_CONNECTION_NAME'), workspace=ws.name)


if __name__ == '__main__':
    set_azure_artifacts_connection()