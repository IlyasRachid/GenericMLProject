import os
import sys

import numpy as np
import pandas as pd

from src.exception import CustomException
import dill
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, 'wb') as file_obj:
                dill.dump(obj, file_obj)
    except Exception as e:
         raise CustomException(e, sys)
    
def evaluate_model(X_train, y_train, X_test, y_test, model, param):
    try:
        report = {}
        for i in range(len(list(model))):
            model_name = list(model.values())[i]
            para = param[list(model.keys())[i]]
            gs = GridSearchCV(model_name, para, cv=3)
            gs.fit(X_train, y_train)
            model_name.set_params(**gs.best_params_)
            model_name.fit(X_train, y_train)
            #model_name.fit(X_train, y_train)
            y_test_pred = model_name.predict(X_test)
            test_model_score = r2_score(y_test, y_test_pred)
            report[list(model.keys())[i]] = test_model_score
        return report

    except Exception as e:
        raise CustomException(e, sys)    
    


def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return dill.load(file_obj)
    except Exception as e:
        raise CustomException(e, sys)