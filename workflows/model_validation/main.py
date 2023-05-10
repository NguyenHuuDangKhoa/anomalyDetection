"""
Through validate_model, the user can validate if their model is ready for registration or not.
There are two validate_model functions, one is validate_model_classification which returns metrics for classification approach.
The other function, validate_model_regression, returns metrics relevent for the NBM_RUL approach.
The validate_model would return True for "ready to register" or False for "skip registration".
The function should also return a logger that has logged anything that went wrong.
"""

from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Any, List, Text, Tuple

import matplotlib.pyplot as plt
import pandas as pd
import structlog

from sklearn.metrics import mean_squared_error
from project_core.model.utils import load_model
from project_core.model.utils import display_time
from project_core.reporting.utils import get_style_file_path
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.preprocessing import MinMaxScaler

plt.style.use(get_style_file_path())

logger = structlog.getLogger(__name__)


def parse_arguments(arguments_list: List) -> Namespace:
    """Parse arguments list"""
    parser = ArgumentParser(__name__)

    parser.add_argument('--input_train_data_path', type=Path, help='Training data path')
    parser.add_argument('--input_X_train_data_file_name', type=Text, help='X Training data file name')
    parser.add_argument('--input_data_path', type=Path, help='path to test input data')
    parser.add_argument('--input_test_data_path', type=Path, help='Testing data path')
    parser.add_argument('--input_X_test_data_file_name', type=Text, help='X Testing data file name')
    parser.add_argument('--input_y_test_data_file_name', type=Text, help='y Testing data file name')
    parser.add_argument('--input_model_path', type=Path, help='path to model directory')
    parser.add_argument('--output_data_path', type=Path, help='output path to write validation results to')
    parser.add_argument('--config_path', type=Path, help='Path to model validation config file')
    parser.add_argument('--model_option', type=Text,
                        choices=['one_class_classification', 'NBM_RUL'],
                        help='Model to be trained')

    return parser.parse_args(arguments_list)


def validate_model_classification(model: Any,
                   X_train: pd.DataFrame,
                   X_test: pd.DataFrame,
                   y_test: pd.DataFrame,
                   save_path: Path) -> Any:
    """Validates the model using test data. Logs results to the run context.
    :param model: a trained model
    :param X_train: input X data for training
    :param X_test: input X data for testing
    :param y_test: target y data for testing
    :param save_path: directory to save the result
    :return: boolean indicating if the model is validated and prediction results
    """
    min_max_scaler = MinMaxScaler()
    min_max_scaler.fit(X_train)
    X_test = min_max_scaler.transform(X_test)
    y_test_pred = model.predict(X_test)
    logger.info(classification_report(y_test, y_test_pred, labels=[1, -1]))
    
    y_test_pred_prob = model.score_samples(X_test)
    roc_auc = roc_auc_score(y_test, y_test_pred_prob)
    logger.info(f'ROC-AUC Score: {roc_auc}')
    
    plt.plot([0, 1], [0, 1], linestyle='--', label='No Skill')
    fpr_negative, tpr_negative, _ = roc_curve(y_test, y_test_pred_prob, pos_label=1)
    fpr_positive, tpr_positive, _ = roc_curve(y_test, -y_test_pred_prob, pos_label=-1)
    plt.plot(fpr_positive, tpr_positive, 'green', lw=1, label='Positive Class')
    plt.plot(fpr_negative, tpr_negative, 'red', lw=1, label='Negative Class')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend()
    plt.title('ROC Curve')
    plt.savefig(save_path)
    return True, pd.DataFrame({'predictions': model.predict(X_test)}), roc_auc

def validate_model_regression(model: Any,
                    X_test: pd.DataFrame, 
                    y_test: pd.DataFrame,
                    rul: bool) -> Tuple[bool, pd.DataFrame]:
    """Validates the model using test data. Logs results to the run context.

    :param model: a trained model
    :param X_test: input data used to validate the NBM
    :param y_test: data used to validate the NBM and compare model predictions to
    :param rul: boolean value, used to display error in a digestable manner if validating rul model
    :return: boolean indicating if the model is validated and prediction results
    """
    score = model.score(X_test, y_test)
    logger.info(f'Model R2 Score: {score}')
    y_pred = model.predict(X_test)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    if rul == False:
        logger.info(f'Model RMSE: {rmse}')
        return True, pd.DataFrame(model.predict(X_test),columns=['0','1','2','3'])
    if rul == True:
        logger.info(f'Model RMSE: {rmse}')
        dated_error = display_time(rmse, 3)
        logger.info(f'RUL error: {dated_error}')
        return True, pd.DataFrame({'predictions': model.predict(X_test)})


def load_datasets(X_test_path: Path, y_test_path: Path, X_train_path: Path = None):
    logger.info('Load y_test')
    y_test = pd.read_feather(path=y_test_path)
    y_test.drop('index',axis=1, inplace=True)

    logger.info('Load X_test')
    X_test = pd.read_feather(path=X_test_path)
    X_test.drop('index',axis=1, inplace=True)

    return X_test, y_test

def run(arguments_list: List = None):
    """Run step"""
    arguments = parse_arguments(arguments_list=arguments_list)

    if arguments.model_option == 'one_class_classification':
        X_train, X_test, y_test = load_datasets(X_test_path=f'{arguments.input_data_path}/{arguments.input_X_test_data_file_name}',
                                            y_test_path=f'{arguments.input_data_path}/{arguments.input_y_test_data_file_name}',
                                            X_train_path=f'{arguments.input_train_data_path}/{arguments.input_X_train_data_file_name}')

        logger.info('Load model')
        model = load_model(model_path=arguments.input_model_path / arguments.input_model_filename)

        logger.info('Validate model')
        result, predictions, roc_auc = validate_model_classification(model=model,
                                                  X_train=X_train,
                                                  X_test=X_test,
                                                  y_test=y_test,
                                                  save_path=f'{arguments.output_data_path}/ROC.png')
        logger.info('Save evaluation result')

        predictions.to_feather(path=arguments.output_data_path / arguments.output_data_file_name)

        if result:
            logger.info('Model validation successful')
        else:
            raise Exception('Model validation failed without an error')
    elif arguments.model_option == 'NBM_RUL':
        #TODO
        pass


if __name__ == '__main__':
    run()