
from datapreprocessor import DataPreProcessor
from data_pipeline import DataPipeline

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


    def choose(self, X):

        raise NotImplementedError

    def updateParameters(self, X, action, reward):

        raise NotImplementedError


class ContextualRandomForestSLPolicy(ContextualPolicy):

    def __init__(self, data):
        self.data_prepocessor = DataPipeline()
        self.X_train, self.X_val, self.y_train, self.y_val = data[0], data[1], data[2], data[3]
        self.RF_model = RandomForestClassifier(random_state=1,
                                          verbose=0)

        param_grid = {
            'n_estimators': [50, 100, 150],
            'max_depth': [5, 6, 7],
        }

        RF_model = RandomForestClassifier(random_state=1,
                                          verbose=0)

        RF_Tuned = GridSearchCV(estimator=RF_model, param_grid=param_grid, cv=StratifiedKFold(3))
        RF_Tuned.fit(self.X_train, self.y_train)

        self.RF_model = RF_Tuned

    def choose(self, X):
        prediction = self.RF_model.predict(X.reshape(1,X.shape[0]))
        return int(prediction)

    def updateParameters(self, X, action, reward):
        pass


class ContextualLinearUCBPolicy(ContextualPolicy):
    def __init__(self, features_size, num_actions=3):
        self.alpha = 0.5
        self.num_actions = num_actions
        self.features_size = features_size
        self.theta = np.zeros((self.num_actions, self.features_size))
        self.A = [np.eye(self.features_size) for _ in range(self.num_actions)]
        self.b = [np.zeros((self.features_size, 1)) for _ in range(self.num_actions)]

    def choose(self, X):
        p = []
        for action in range(self.num_actions):
            A_a_inv = np.linalg.inv(self.A[action])
            theta_hat = A_a_inv.dot(self.b[action])
            p_t_a = theta_hat.T.dot(X) + self.alpha * np.sqrt(X.T.dot(A_a_inv).dot(X))
            p.append(p_t_a)

        a_star = np.argmax(p)
        return a_star

    def updateParameters(self, X, action, reward):
        self.A[action] += np.dot(X, X.T)
        self.b[action] += reward * X.reshape(X.shape[0],1)