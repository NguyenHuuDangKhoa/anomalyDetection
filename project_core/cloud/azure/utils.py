"""
This file includes helper methods for Azure Machine Learning.
"""

from typing import List, Optional, Text

import structlog
from azureml.core import Dataset, Datastore, Experiment, Run, Workspace
from azureml.core.compute import ComputeTarget
from azureml.core.compute_target import ComputeTargetException
from azureml.core.runconfig import RunConfiguration
from azureml.data import FileDataset
from azureml.data.azure_storage_datastore import AzureBlobDatastore
from azureml.pipeline.core import Pipeline, PipelineData
from azureml.pipeline.core.builder import PipelineStep

logger = structlog.getLogger(__name__)


def get_aml_workspace(config_file_location: Optional[str] = None) -> Workspace:
    """
    Returns a workspace object.

    :param config_file_location: path to the config.json file for this Workspace.
    :return ws: Azure ML Workspace
    """
    ws = Workspace.from_config(path=config_file_location)
    return ws


def get_amlcompute_cluster(
    workspace: Workspace,
    cluster_name: str,
) -> ComputeTarget:
    """
    Retrieves Azure Machine Learning compute cluster.

    :param workspace: Azure Workspace object
    :param cluster_name: the given name of the cluster
    :return amlcompute: Azure Compute Target
    """

    # Verify that cluster does not exist already
    try:
        amlcompute = ComputeTarget(workspace=workspace, name=cluster_name)
    except ComputeTargetException:
        logger.error(f'Couldn\'t find the specified compute cluster {cluster_name}, please check if it has been added\
            on AzureML\'s Compute interface')

    amlcompute.wait_for_completion(show_output=True)

    return amlcompute


def create_aml_run_config(
    aml_compute: ComputeTarget,
    config_name: str,
    conda_environment_file: str,
    core_package_name: Text,
    core_package_version: Text,
    core_package_url: Text,
) -> RunConfiguration:
    """
    Creates an Azure Machine Learning run config.
    :param aml_compute: Azure Compute Cluster.
    :param core_package_name: Name of core package
    :param core_package_version: Version of core package
    :param core_package_url: Core package repository URL (starts with --extra-index-url)
    :return aml_run_config: Azure Azure Machine Learning RunConfiguration
    """
    # Create a new runconfig object
    aml_run_config = RunConfiguration()

    # Use the aml_compute you created above.
    aml_run_config.target = aml_compute

    # Enable Docker
    aml_run_config.environment.docker.enabled = True

    # Set Docker base image to the default CPU-based image
    aml_run_config.environment.docker.base_image = 'mcr.microsoft.com/azureml/base:0.2.1'

    # Use conda_dependencies.yaml to create a conda environment in the Docker image for execution
    aml_run_config.environment.python.user_managed_dependencies = False

    # Specify CondaDependencies obj, add necessary packages
    aml_run_config.environment = aml_run_config.environment.from_conda_specification(
        name=config_name,
        file_path=conda_environment_file,
    )

    core_package = core_package_name + '==' + core_package_version
    aml_run_config.environment.python.conda_dependencies.add_pip_package(core_package)

    # Add core package url
    aml_run_config.environment.python.conda_dependencies.set_pip_option(core_package_url)

    return aml_run_config


def get_aml_datastore(workspace: Workspace, name: Optional[str]) -> AzureBlobDatastore:
    """
    Retrieve default or a named Azure Machine Learning Datastore registered with a workspace.
    :param workspace: Azure Machine Learning Workspace
    :return datastore: Azure Machine Learning Datastore
    """
    if name:
        datastore = Datastore.get(workspace, name)
    else:
        datastore = workspace.get_default_datastore()

    return datastore


def create_aml_pipeline_data(name: str, datastore: Datastore) -> PipelineData:
    """
    Create Azure Machine Learning Data Pipline.
    :param name: Data pipeline name
    :param datastore: Azure Machine Learning Datastore
    :return data_pipeline: Azure Machine Learning data pipeline
    """
    data_pipeline = PipelineData(name=name, datastore=datastore).as_dataset()
    return data_pipeline


def _create_aml_experiment(workspace: Workspace, name: str) -> Experiment:
    """
    Create an Azure Machine Learning Experiment.
    :param workspace: Azure Workspace object
    :param name: experiment name
    :return experiment: Azure Machine Learning Experiment
    """
    experiment = Experiment(workspace, name=name)

    return experiment


def aml_submit_experiment(workspace: Workspace, name: str, pipeline_steps: List[PipelineStep]) -> Run:
    """
    Submit a pipeline for execution on the Azure Compute cluster.

    :param workspace: Azure Machine Learning Workspace
    :param name: Azure Experiment name
    :param pipeline_steps: List of pipeline steps
    """
    experiment = _create_aml_experiment(workspace=workspace, name=name)
    pipeline = Pipeline(workspace=workspace, steps=pipeline_steps)
    experiment_run = experiment.submit(pipeline, regenerate_outputs=False)

    return experiment_run


def get_aml_dataset(
    workspace: Workspace,
    dataset_name: str,
    dataset_version: int = 'latest'
) -> Dataset:
    """
    Get the tabular dataset registered in the workspace. If not found, return None.

    :param workspace: Azure Workspace
    :param dataset_name: dataset name
    :param dataset_version: dataset version, defaults to 'latest'
    :return: Dataset object
    """
    try:
        data_set = Dataset.get_by_name(workspace=workspace, name=dataset_name, version=dataset_version)
        logger.info('Found registered dataset.')
    except Exception as ex:
        logger.error('Dataset not found, please make sure to register it in the right workspace')
        logger.error(ex)
        raise
    return data_set


def register_aml_dataset(
    workspace: Workspace,
    datastore: Datastore,
    file_path: str,
    dataset_name: str,
) -> FileDataset:
    """
    Register a dataset in the workspace

    :param workspace: Azure Workspace
    :param datastore: Azure Datastore
    :param file_path: dataset file path
    :param dataset_name: dataset name
    :return: Dataset object
    """
    try:
        # FIXME
        data_set = Dataset.File.from_files(path=(datastore, file_path))
        data_set = data_set.register(
            workspace=workspace,
            name=dataset_name,
            tags={'format': 'CSV'},
            create_new_version=True,
        )
        logger.info('New dataset registered.')
    except Exception as ex:
        logger.error('Unable to register the dataset')
        logger.error(ex)
        raise

    return data_set
