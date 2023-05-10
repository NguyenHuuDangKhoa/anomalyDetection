## Description
Retrieve SCADA data of "equipment" from Azure Blob Storage then run a pipeline to train a machine learning model to detect anomalies for predictive maintenance


## Setup
We use a Conda environment file to manage dependencies and unify our virtual environments. To create the environment,
change the environment name in [environment.yaml](environment.yaml) file to a unique name for this project / use-case.

>Note 1: A Conda best practice states, Conda works best with installing all required dependencies while building the
environment. So, whenever you add a new library to the environment, you should delete, and rebuild the environment.
The effect of this can be that new installations take a long time to resolve conflicts.


However, Conda as a package manager is infamous for being slow when solving environment conflicts. In other words, it takes a lot of time determining which versions of different packages can work together. Therefore, it is recommended to use Mamba as an alternative package manager.


>Note 2: More information of Mamba can be found here [Mamba's Documentation]

To setup Mamba, run the following command:

```bash
make pre-setup-environment
```

>Note 3: This is a one-time process and the above command only needs to be run once when a new compute instance is created.

Then, every time you need to create a new environment, just run the following command: 

```bash
make setup-environment
```

>Note 4: Mamba is used only for creating new environments. Other tasks, e.g. activate a environment, can be done through Conda.

## Important Note

All of the restricted information, but are required for the pipeline to run, are stored in the .env file (the . before env is important). Please make sure to create a new .env file after cloning the repo.

### Acquire Data

Data acquisition code, such as

- Download data from blob storage
- Query a database to get the dataset file

Should go into [workflows/data_acquisition](./workflows/data_acquisition/main.py)

What gets put into that script should be runnable from the root directory using

```bash
make acquire-data
```

### Pipeline

The pipeline can be run using this command. You can go ahead and try it

```bash
make run-pipeline
```

<!-- links -->
[Mamba's Documentation]: https://mamba.readthedocs.io/en/latest/index.html
[Create Azure Machine Learning datasets]: https://docs.microsoft.com/en-us/azure/machine-learning/how-to-create-register-datasets
[TabularDatasetFactory Class]: https://docs.microsoft.com/en-us/python/api/azureml-core/azureml.data.dataset_factory.tabulardatasetfactory?view=azure-ml-py
[Dataset Class]: https://docs.microsoft.com/en-us/python/api/azureml-core/azureml.core.dataset.dataset?view=azure-ml-py
[ipython documentation]: https://ipython.readthedocs.io/en/stable/install/kernel_install.html#kernels-for-different-environments