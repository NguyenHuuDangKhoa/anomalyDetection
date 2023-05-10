"""
This file creates a pipeline to run on Azure.
"""
from typing import Dict, Text

import structlog
from azureml.data import OutputFileDatasetConfig
from azureml.pipeline.core import PortDataReference, StepRun
from azureml.pipeline.steps import PythonScriptStep

#FIXME
from project_core.cloud.azure.utils import (aml_submit_experiment,
                                            create_aml_run_config,
                                            get_aml_dataset, get_aml_datastore,
                                            get_aml_workspace,
                                            get_amlcompute_cluster)

logger = structlog.getLogger(__name__)

#FIXME
def run(run_config: Dict, experiment_output_path: Text):
    """
    This method creates a pipeline consisted of Data Acquisition,
    Data Processing, and Model Training steps and submits it to azure
    compute cluster to run. The trained model artifact and run info
    is saved in the output directory.

    :param run_config: Configuration for running the scripts
    """
    output_file_dataset_destination = experiment_output_path + '/{output-name}'

    # Get Azure Machine Learning Workspace object
    aml_ws = get_aml_workspace()

    # Define Azure Machine Learning Compute CLuster
    aml_compute = get_amlcompute_cluster(
        workspace=aml_ws,
        cluster_name=run_config.get('default_azure_configs').get('compute_cluster_name'))

    aml_run_config = create_aml_run_config(
        aml_compute=aml_compute,
        config_name=run_config.get('default_azure_configs').get('run_config_name'),
        conda_environment_file=run_config.get('default_azure_configs').get('conda_environment_file'),
        core_package_name=run_config.get('default_azure_configs').get('package').get('name'),
        core_package_version=run_config.get('default_azure_configs').get('package').get('version'),
        core_package_url=run_config.get('default_azure_configs').get('package').get('url'),
    )

    aml_datastore = get_aml_datastore(
        workspace=aml_ws,
        name=run_config.get('default_azure_configs').get('datastore_name'),
    )

    aml_dataset = get_aml_dataset(
        workspace=aml_ws,
        dataset_name=run_config.get('default_azure_configs').get('dataset_name'),
        dataset_version=run_config.get('default_azure_configs').get('dataset_version'),
    )

    def get_step_configs(step): return run_config.get('steps_to_run').get(step)
    # Define Data validation Step
    data_validation_configs = get_step_configs('data_validation')
    data_validation_step = PythonScriptStep(
        name=data_validation_configs.get('name'),
        script_name=data_validation_configs.get('code').get('script_name'),
        arguments=[
            '--input_data_path', aml_dataset.as_named_input('raw_data').as_download(),
            *data_validation_configs.get('arguments').get('azure')],
        compute_target=aml_compute,
        runconfig=aml_run_config,
        source_directory=f'{run_config.get("source_directory_root")}/{data_validation_configs.get("code").get("source_path")}',
        allow_reuse=True,
    )

    # Define the Data Pipeline for Train Data
    train_data_config = OutputFileDatasetConfig(
        name=run_config.get('train_data_pipeline_name'),
        destination=(aml_datastore, output_file_dataset_destination),
    )
    # Define the Data Pipeline for Test Data
    test_data_config = OutputFileDatasetConfig(
        name=run_config.get('test_data_pipeline_name'),
        destination=(aml_datastore, output_file_dataset_destination),
    )

    # Define the Processing Data Step
    data_processing_configs = get_step_configs('data_processing')
    data_processing_step = PythonScriptStep(
        name=data_processing_configs.get('name'),
        script_name=data_processing_configs.get('code').get('script_name'),
        arguments=[
            '--input_data_path', aml_dataset.as_named_input('raw_data').as_download(),
            '--output_train_data_path', train_data_config,
            '--output_test_data_path', test_data_config,
            *data_processing_configs.get('arguments').get('azure')],
        compute_target=aml_compute,
        runconfig=aml_run_config,
        source_directory=f'{run_config.get("source_directory_root")}/{data_processing_configs.get("code").get("source_path")}',
        allow_reuse=True,
    )

    data_processing_step.run_after(data_validation_step)

    # Define data pipeline to save model after training step

    models_data_config = OutputFileDatasetConfig(
        name=run_config.get('model_data_pipeline_name'),
        source=run_config.get('default_azure_configs').get('outputs_directory'),
        destination=(aml_datastore, output_file_dataset_destination),
    )

    model_training_configs = get_step_configs('model_training')
    # Define the Model Training Step
    model_training_step = PythonScriptStep(
        name=model_training_configs.get('name'),
        script_name=model_training_configs.get('code').get('script_name'),
        arguments=[
            '--input_train_data_path', train_data_config.as_input(),
            '--input_test_data_path', test_data_config.as_input(),
            '--output_model_path', run_config.get('default_azure_configs').get('outputs_directory'),
            *model_training_configs.get('arguments').get('azure')],
        outputs=[models_data_config],  # This is needed for Azure to consider the ./outputs path an output of this step
        compute_target=aml_compute,
        runconfig=aml_run_config,
        source_directory=f'{run_config.get("source_directory_root")}/{model_training_configs.get("code").get("source_path")}',
        allow_reuse=True,
    )

    model_validation_configs = get_step_configs('model_validation')

    model_validation_data_config = OutputFileDatasetConfig(
        name=run_config.get('model_validation_data_pipeline_name'),
        source=run_config.get('default_azure_configs').get('outputs_directory'),
        destination=(aml_datastore, output_file_dataset_destination),
    )

    # Define the Model Validation Step
    model_validation_step = PythonScriptStep(
        name=model_validation_configs.get('name'),
        script_name=model_validation_configs.get('code').get('script_name'),
        arguments=[
            '--input_data_path', test_data_config.as_input(),
            '--input_model_path', models_data_config.as_input(),
            '--output_data_path', run_config.get('default_azure_configs').get('outputs_directory'),
            *model_validation_configs.get('arguments').get('azure')],
        outputs=[model_validation_data_config],
        compute_target=aml_compute,
        runconfig=aml_run_config,
        source_directory=f'{run_config.get("source_directory_root")}/{model_validation_configs.get("code").get("source_path")}',
        allow_reuse=True,
    )

    model_validation_step.run_after(model_training_step)
    # Define the experiment pipeline. Note that only running
    # the last step would be enough. This is because, Azure runs
    # the previous steps that are required for running each step.
    aml_experiment_pipeline_steps = [
        data_validation_step,
        data_processing_step,
        model_training_step,
        model_validation_step,
    ]

    # Submit the Experiment to run on Azure compute cluster.
    aml_pipeline_run = aml_submit_experiment(
        workspace=aml_ws,
        name=run_config.get('default_azure_configs').get('experiment_name'),
        pipeline_steps=aml_experiment_pipeline_steps,
    )
    logger.info('Pipeline submitted for execution.')

    # Wait for the experiment pipeline to complete.
    pipeline_status = aml_pipeline_run.wait_for_completion(show_output=False)
    logger.info(f'Pipeline completed with status: {pipeline_status}.')

    run: StepRun = aml_pipeline_run.find_step_run(model_training_configs.get('name'))[0]

    logger.info('Downloading model')
    model: PortDataReference = run.get_output_data(
        name=run_config.get('model_data_pipeline_name'),
    )
    model.download(local_path=run_config.get('output_directory'))

    run: StepRun = aml_pipeline_run.find_step_run(model_validation_configs.get('name'))[0]

    logger.info('Downloading validation')
    data: PortDataReference = run.get_output_data(
        name=run_config.get('model_validation_data_pipeline_name'),
    )
    data.download(local_path=run_config.get('output_directory'))

    return aml_pipeline_run
