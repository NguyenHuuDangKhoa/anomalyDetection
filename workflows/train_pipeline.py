"""
Runs the Azure pipeline defined in `azure/` or
individually imports each module of the pipeline and runs locally
"""
from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path
from typing import Dict, Text

import structlog
from data_processing.main import main as data_processing_run
from data_validation.main import run as data_validation_run
from model_training.main import run as model_training_run
from model_validation.main import run as model_validation_run

from project_core.utils import load_yaml_config

logger = structlog.getLogger(__name__)

parser = ArgumentParser(__name__)
parser.add_argument('--config_path', type=Path, help='Path of the pipeline config file')


def local_run(run_config: Dict, experiment_output_path: Text):
    """
    This method runs a train pipeline on the compute instance.

    :param run_config: Configuration for running the scripts
    :param experiment_output_path: String of path to the experiment log directory (should not have a slash / at the end)
    """
    logger.info('In train_pipeline.py -> Run local_pipeline.py')
    local_pipeline.run(run_config=run_config, experiment_output_path=experiment_output_path)


def azure_run(run_config: Dict, experiment_output_path: Text):
    """
    This method runs a train pipeline on azure compute cluster

    :param run_config: Configuration for running the scripts
    :param experiment_output_path: String of path to the experiment log directory (should not have a slash / at the end)
    """
    azure_pipeline.run(run_config=run_config, experiment_output_path=experiment_output_path)


if __name__ == '__main__':
    arguments = parser.parse_args()
    run_config = load_yaml_config(config_path=arguments.config_path)

    starting_timestamp = datetime.isoformat(datetime.utcnow())

    if run_config.get('compute') == 'local':
        # Create output directory in root folder
        experiment_output_path = f'{run_config.get("output_directory")}/run-logs/{starting_timestamp}'
        import local_pipeline
        local_run(run_config=run_config, experiment_output_path=experiment_output_path)
        pass
    elif run_config.get('compute') == 'azure':
        #TODO
        pass
