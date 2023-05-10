"""
This file creates a pipeline to run on the compute instance.
"""
from pathlib import Path
from typing import Dict, Text

import structlog
from data_processing.main import main as data_processing_run
from data_validation.main import run as data_validation_run
from model_training.main import run as model_training_run
from model_validation.main import run as model_validation_run

logger = structlog.getLogger(__name__)


def run(run_config: Dict, experiment_output_path: Text):
    """
    This method creates a pipeline consisted of Data Processing, and Model Training steps and runs it on the
    compute instance. The trained model artifact and run info are saved in the output directory.

    :param run_config: Configuration for running the scripts
    :param experiment_output_path: String of path to the experiment log directory (should not have a slash / at the end)
    """
    output_file_dataset_destination = experiment_output_path + '/{output_name}'

    step_run_reference = {
        'data_validation': data_validation_run,
        'data_processing': data_processing_run,
        'model_training': model_training_run,
        'model_validation': model_validation_run,
    }

    def get_step_configs(step): return run_config.get('steps_to_run').get(step)
    def make_directory(path_string): return Path(path_string).mkdir(parents=True, exist_ok=True)

    # Define Data validation Step
    step_name = 'data_validation'
    step_configs = get_step_configs(step_name)
    if step_configs.get('run'):
        logger.info(f'Running step: {step_name}')
        step_arguments = step_configs.get('arguments').get('local')
        step_run_reference.get(step_name)(arguments_list=step_arguments)
    else:
        logger.warn(f'{step_name} run config set to: {step_configs.get("run")}')

    # Define the Data Pipeline for Train Data
    train_data_config = output_file_dataset_destination.format(
        output_name=run_config.get('train_data_pipeline_name'),
    )
    make_directory(train_data_config)
    # Define the Data Pipeline for Test Data
    test_data_config = output_file_dataset_destination.format(
        output_name=run_config.get('test_data_pipeline_name'),
    )
    make_directory(test_data_config)

    # Define Data processing Step
    logger.info('In local_pipeline.py')
    step_name = 'data_processing'
    step_configs = get_step_configs(step_name)
    if step_configs.get('run'):
        logger.info(f'Running step: {step_name}')
        step_arguments = [
            '--output_train_data_path', train_data_config,
            '--output_test_data_path', test_data_config,
            *step_configs.get('arguments').get('local'),  # Packing the arguments to a List
        ]
        step_run_reference.get(step_name)(arguments_list=step_arguments)
        logger.info('In local_pipeline.py -> Run data_processing.main.py')
    else:
        logger.warn(f'{step_name} run config set to: {step_configs.get("run")}')

    # Define data pipeline to save model after training step
    models_data_config = output_file_dataset_destination.format(
        output_name=run_config.get('model_data_pipeline_name'),
    )
    make_directory(models_data_config)

    # Define Model training Step
    step_name = 'model_training'
    step_configs = get_step_configs(step_name)
    if step_configs.get('run'):
        logger.info(f'Running step: {step_name}')
        step_arguments = [
            '--input_train_data_path', train_data_config,
            '--input_test_data_path', test_data_config,
            '--output_model_path', models_data_config,
            *step_configs.get('arguments').get('local'),
        ]
        step_run_reference.get(step_name)(arguments_list=step_arguments)
    else:
        logger.warn(f'{step_name} run config set to: {step_configs.get("run")}')

    model_validation_data_config = output_file_dataset_destination.format(
        output_name=run_config.get('model_validation_data_pipeline_name'),
    )
    make_directory(model_validation_data_config)

    # Define Model validation Step
    step_name = 'model_validation'
    step_configs = get_step_configs(step_name)
    if step_configs.get('run'):
        logger.info(f'Running step: {step_name}')
        step_arguments = [
            '--input_data_path', test_data_config,
            '--input_model_path', models_data_config,
            '--output_data_path', model_validation_data_config,
            *step_configs.get('arguments').get('local'),
        ]
        step_run_reference.get(step_name)(arguments_list=step_arguments)
    else:
        logger.warn(f'{step_name} run config set to: {step_configs.get("run")}')
