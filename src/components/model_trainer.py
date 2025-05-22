import os
import sys
from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging

from src.utils import save_object
from src.utils import evaluate_model

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifacts', 'model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Splitting training and test data")
            X_train, y_train, X_test, y_test = (
                train_array[:,:-1], train_array[:, -1], test_array[:, :-1], test_array[:,-1]
            )
            models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "K-Neighbors Classifier": KNeighborsRegressor(),
                "XGBClassifier": XGBRegressor(),
                "CatBoosting Classifier": CatBoostRegressor(verbose=False),
                "AdaBoost Classifier": AdaBoostRegressor()
            }

            params = {
                "Decision Tree": {
                    'criterion': ['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    #'splitter': ['best', 'random'],
                    #'max_depth': [None, 10, 20, 30, 40, 50],
                    #'max_features': ['auto', 'sqrt', 'log2'],
                },
                "Random Forest": {
                    'n_estimators': [8,16,32,64,128,256],
                    #'criterion': ['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                },
                "Gradient Boosting": {
                    'n_estimators': [8,16,32,64,128,256],
                    'learning_rate': [.1, .01, .05, .001],
                    "subsample": [0.6, 0.7, 0.8, 0.85, 0.9],
                    #'max_depth': [3, 5, 7],
                    #'min_samples_split': [2, 5, 10],
                    #'min_samples_leaf': [1, 2, 4],
                },
                "Linear Regression": {},
                "K-Neighbors Classifier": {
                    'n_neighbors': [5, 7, 9, 11],
                    #'weights': ['uniform', 'distance'],
                    #'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
                    #'leaf_size': [10, 20, 30],
                },
                "XGBClassifier": {
                    'n_estimators': [8,16,32,64,128,256],
                    'learning_rate': [.1, .01, .05, .001],
                    #'max_depth': [3, 5, 7],
                    #'min_child_weight': [1, 2, 3],
                    #'subsample': [0.6, 0.7, 0.8, 0.85, 0.9],
                },
                "CatBoosting Classifier": {
                    'iterations': [30,50,100],
                    'learning_rate': [.1, .01, .05, .001],
                    'depth': [6,8,10],
                    #'l2_leaf_reg': [1, 3, 5],
                },
                "AdaBoost Classifier": {
                    'n_estimators': [8,16,32,64,128,256],
                    'learning_rate': [.1, .01, .05, .001],
                    #'base_estimator': [None, DecisionTreeRegressor(max_depth=3)],
                }

            }

            model_report: dict= evaluate_model(X_train=X_train,y_train=y_train,X_test=X_test, y_test=y_test, model=models, param=params)
            best_model_score = max(sorted(model_report.values()))
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]

            best_model = models[best_model_name]
            if best_model_score < 0.6:
                raise CustomException("No best model found")
            logging.info("Best model found")
            save_object(
                file_path= self.model_trainer_config.trained_model_file_path,
                obj= best_model
            )
            logging.info("Model saved")
            
            return best_model_name, best_model_score
        
        except Exception as e:
            logging.info("Error in model trainer")
            raise CustomException(e, sys)
            
    