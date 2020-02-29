
from .datapreprocessor import DataPreProcessor
import pandas as pd
import numpy as np

from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

import statsmodels.api as sm
from sklearn.feature_selection import RFECV
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.linear_model import LinearRegression, LassoCV, RidgeCV
from sklearn.feature_selection import SelectFromModel

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler

class ContextualPolicy(object):
    """
    Abstract class for a contextual multi-armed bandit policy/algorithm
    """

    def __init__(self):
        raise NotImplementedError


    def choose(self, X, eval=False):

        raise NotImplementedError

    def updateParameters(self, X, action, reward):

        raise NotImplementedError


class ContextualRandomForestSLPolicy(ContextualPolicy):

    def __init__(self):
        self.data_prepocessor = DataPreProcessor()
        self.X_train, self.X_val, self.y_train, self.y_val = self.data_prepocessor.loadAndPrepData()
        self.RF_model = RandomForestClassifier(random_state=1,
                                          verbose=0)

        param_grid = {
            'n_estimators': [50, 100, 150],
            'max_depth': [5, 6, 7],
        }

        RF_model = RandomForestRegressor(random_state=1,
                                          verbose=0)  ### we can change that to regression once we discritize the labels

        RF_Tuned = GridSearchCV(estimator=RF_model, param_grid=param_grid, cv=StratifiedKFold(3))
        RF_Tuned.fit(self.X_train, self.y_train)

        self.RF_model = RF_Tuned

    def choose(self, X, eval=False):
        prediction = self.RF_model.predict(X)
        return prediction

    def updateParameters(self, X, action, reward):
        pass