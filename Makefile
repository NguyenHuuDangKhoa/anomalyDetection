MAGENTA=`tput setaf 5`
RESET=`tput sgr0`

all: acquire-data run-pipeline

acquire-data:
	@python workflows/data_acquisition/main.py \
		--data All_Sites_Major_Component_Tracking__xlsx gear_failure__csv generator_failure__csv grd_data__feather tmp_data__feather tur_data__feather\
		--config_file_name config_data_acquisition.yaml \
		--action download_blob \

register-azure-dataset:
	@echo "Registering dataset on AzureML"
	@python workflows/register_azure_dataset.py --config_path=workflows/configs/run_config.yaml

register-model:
	@echo "Register model"
	@echo "${MAGENTA}FIXME need to change the demo model path to fit what you're working on${RESET}"
	python workflows/model_registration/main.py \
		--input_model_path=./data/model/ \
		--config_path=config.yaml

pre-setup-environment:
	@echo "${MAGENTA}Activate base environment if run into any issue${RESET}"
	@conda config --add channels conda-forge
	@conda install mamba -n base -c conda-forge

setup-environment:
	@echo "Setup world"
	@conda config --set channel_priority strict
	@mamba env create -f environment.yaml
	@echo "${MAGENTA}Remember to activate your environment with these instructions ^${RESET}"

setup-package:
	@echo "Installing project_core in development mode with ${MAGENTA}pip install -e .${RESET}"
	@echo "Now you can use the current state of code in project_core anywhere in this environment"
	pip install -e .

run-pipeline:
	@echo "Running pipeline"
	@python workflows/train_pipeline.py --config_path=workflows/configs/run_config.yaml

set-azure-artifacts-connection:
	@echo "${MAGENTA}Setting up connection${RESET}"
	@python workflows/packaging/set_azure_artifacts_connection.py --config_path=config.yaml

format:
	@echo "Running autopep8 and isort to fix any formatting issues in the code"
	@autopep8 --in-place --recursive .
	@isort .

build-package:
	python -m build --wheel
