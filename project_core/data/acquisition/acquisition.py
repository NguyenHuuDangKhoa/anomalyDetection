"""A class that is responsible for data acquisition"""
import os
from azure.storage.blob import BlobServiceClient
import time
import mltable
import pandas as pd
from pathlib import Path
import structlog

logger = structlog.getLogger(__name__)


def make_dirs(path: Path, file_name: str):
    """
    This functions create directories provided in a path, together with a timestamp and file name
    :param path: a path that contains directories to be created; should use double back slashes
    :param file_name: a name of a file
    """
    # Make new directory using timestamp
    path = os.path.join(path, file_name.split('.')[0])
    # os.makedirs is used instead of os.mkdir because there might be many directories provided in a path
    os.makedirs(path, exist_ok=True)
    return path


class Data:
    """
    This class is responsible for operations involve raw data,
    which includes downloading data, converting data, and saving data.
    :attribute uri_type: type of the data to be downloaded, i.e. 'file' or 'folder
    :attribute datastore_uri: location of the data to be downloaded on Azure Blob Storage
    """

    def __init__(self, uri_type: str, datastore_uri: str) -> None:
        self.uri_type = uri_type
        self.datastore_uri = datastore_uri

    def __create_mltable_from_csv(self) -> mltable.MLTable:
        """
        This private method creates a blueprint (a YAML-based file) that
        defines how data should be materialized in Pandas data frame.
        :return: a MLTable file
        """
        path = {self.uri_type: self.datastore_uri}
        logger.info('Creating a MLTable (YAML file) from .csv files...')
        return mltable.from_delimited_files(paths=[path])

    def __convert_mltable_to_pandas_dataframe(self) -> pd.DataFrame:
        """
        This private method convert a MLTable file to a Pandas data frame.
        :return: a Pandas data frame
        """
        data = self.__create_mltable_from_csv()
        logger.info('Converting the MLTable (YAML file) to a Pandas DataFrame...')
        return data.to_pandas_dataframe()

    def _convert_pandas_dataframe_to_feather(self, file_name: str,  path: Path = f'{Path(__file__).parent.parent.parent.parent}/data/raw_data') -> None:
        """
        This protected method convert a Pandas data frame to a .feather file format and
        save to a provided path for storing.
        :param file_name: a name for the .feather file; should in the following format '<data_name>.feather'
        :param path: a path to a location that the .feather file will be saved; should use double back slashes
        """
        data = self.__convert_mltable_to_pandas_dataframe()
        logger.info('Converting the Pandas DataFrame to a .feather file for storing...')

        # make_dirs makes new directories and includes timestamp in the path
        # first file_name is for making directories, which does not have file extension while the last does have one
        path = os.path.join(make_dirs(path=path, file_name=file_name), file_name)

        data.to_feather(path=path)
        logger.info(f'A {file_name} file is created at {path}')


class BlobData(Data):
    """
    This class inherits from the Data class and is responsible for extracting blob names in Azure Data Container,
    as well as downloading specific blob.
    :attribute uri_type: type of the data to be downloaded, i.e. 'file' or 'folder; inherits from Data class
    :attribute datastore_uri: location of the data to be downloaded on Azure Blob Storage; ; inherits from Data class
    :attribute container_name: name of the Azure container that contains a blob to be downloaded
    :attribute blob_name: name of the blob to be downloaded
    :attribute connect_str: a security string needed to connect to a specifc Azure Storage
    """

    def __init__(self, uri_type: str, datastore_uri: str, container_name: str, connect_str: str, blob_name: str = None) -> None:
        super().__init__(uri_type, datastore_uri)
        self.container_name = container_name
        self.blob_name = blob_name
        self.connect_str = connect_str

    def __get_blob_names(self, file_name: str, path: Path = f'{Path(__file__).parent.parent.parent.parent}/data/download/blob_names') -> None:
        """
        This private method gets all names of blobs that are currently stored on the provided Azure container.
        :param file_name: name of a file that stores all the acquired names; format: '<file_name>.txt'
        :param path: a path to a location that the file will be saved; should use double back slashes
        """
        # Create the BlobServiceClient object
        blob_service_client = BlobServiceClient.from_connection_string(self.connect_str)

        # List the blobs in the container
        # Since we want to work with container here, we use get.container_client
        container_client = blob_service_client.get_container_client(self.container_name)
        blob_names = container_client.list_blobs()

        # make_dirs makes new directories and includes timestamp in the path
        # first file_name is for making directories, which does not have file extension while the last does have one
        path = os.path.join(make_dirs(path=path, file_name=file_name), file_name)

        logger.info('Extracting all blob names on Azure Blob Storages...')
        with open(file=path, mode="w") as download_file:
            for blob in blob_names:
                download_file.write(f'{blob.name}\n')  # blob.name get the name attribute of the blob

    def _download_from_blob_storage(self, file_name: str, path: Path = f'{Path(__file__).parent.parent.parent.parent}/data/download/raw_data') -> None:
        """
        This protected method download a specific blob with provided name.
        :param file_name: a name for the file that is downloaded; format: '<data_name>_download.<extension>'
        :param path: a path to a location that the file will be saved; should use double back slashes
        """
        # Create the BlobServiceClient object
        blob_service_client = BlobServiceClient.from_connection_string(self.connect_str)

        # Since we want to work with specific here, we use get.blob_client and provide that blob name
        blob_client = blob_service_client.get_blob_client(container=self.container_name, blob=self.blob_name)
        logger.info(f'Connecting to Azure Blob Storage and downloading {self.blob_name}...')

        path = os.path.join(make_dirs(path=path, file_name=file_name), file_name)

        with open(file=path, mode="wb") as download_file:
            download_file.write(blob_client.download_blob().readall())
        logger.info(f'A {file_name} file is downloaded at {path}')
