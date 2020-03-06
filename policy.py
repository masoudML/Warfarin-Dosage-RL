from data_pipeline import DataPipeline
from sklearn.svm import SVC, LinearSVC
import pandas as pd
import numpy as np

from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression

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
from scipy.stats import beta


class Policy(object):

    def __init__(self):
        raise NotImplementedError


    def choose(self, X):

        raise NotImplementedError

    def updateParameters(self, X, action, reward):

        raise NotImplementedError


class RandomForestSLPolicy(Policy):

    def __init__(self, data):
        self.data_prepocessor = DataPipeline()
        self.X_train, self.X_val, self.y_train, self.y_val = data
        self.RF_model = LogisticRegression(random_state=1,
                                          verbose=0)

        param_grid = {
            'n_estimators': [50, 100, 150, 200],
            'max_depth': [5, 6, 7, 8, 9 , 10, 12, 15],
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


class LogisticRegressionSLPolicy(Policy):

    def __init__(self, data):
        self.data_prepocessor = DataPipeline()
        self.X_train, self.X_val, self.y_train, self.y_val = data
        self.lr_model = LogisticRegression(random_state=1,multi_class='multinomial',solver='newton-cg',
                                          verbose=0)

        self.lr_model.fit(self.X_train, self.y_train)

    def choose(self, X):
        prediction = self.lr_model.predict(X.reshape(1,X.shape[0]))
        return int(prediction)

    def updateParameters(self, X, action, reward):
        pass

class ContextualLinearUCBPolicy(Policy):
    def __init__(self, features_size, num_actions=3):
        self.alpha = 1
        self.delta = 0.05
        self.num_actions = num_actions
        self.features_size = features_size
        self.theta = np.zeros((self.num_actions, self.features_size))
        self.A = [np.identity(self.features_size) for _ in range(self.num_actions)]
        self.b = [np.zeros((self.features_size, 1)) for _ in range(self.num_actions)]
        self.time_step = 0
    def choose(self, X):
        self.time_step += 1
        self.alpha = 1 + np.sqrt(np.log(2*self.time_step/self.delta)/2)

        p = []
        for action in range(self.num_actions):
            A_a_inv = np.linalg.inv(self.A[action])
            theta_hat = A_a_inv.dot(self.b[action])
            p_t_a = theta_hat.T.dot(X) + self.alpha * np.sqrt(X.T.dot(A_a_inv).dot(X))
            p.append(p_t_a)

        a_star = np.argmax(p)
        return a_star

    def updateParameters(self, X, action, reward):
        self.A[action] += np.outer(X, X)
        self.b[action] += reward * X.reshape(X.shape[0],1)


class ThompsonSamplingContextualBanditPolicy(Policy):
    def __init__(self, features_size, num_actions=3):
        ### hyper-parameters
        #### from the paper: v = R . sqrt((24/eps) d log(1/delta))
        self.R = 0.1
        self.eps = 0.01
        self.delta = 0.05
        self.v = 1
        self.num_actions = num_actions
        self.features_size = features_size
        self.B = [np.identity(self.features_size) for _ in range(self.num_actions)]
        self.mu_hat = [np.zeros((self.features_size,1)) for _ in range(self.num_actions)]
        self.f = [np.zeros(self.features_size) for _ in range(self.num_actions)]
        self.time_step = 0
    def choose(self, X):
        self.time_step += 1
        self.v = self.R * np.sqrt(9 * '''self.feature_size''' * np.log(max(1.1,np.log(self.time_step/self.delta)))) ### d feature size can be removed, R is the range but that would be in extract

        p = []
        for action in range(self.num_actions):
            mu_tilda = np.random.multivariate_normal(mean=self.mu_hat[action].reshape(self.features_size),
                                                     cov=((self.v ** 2) * np.linalg.inv(self.B[action])))
            p_t_a = np.dot(X.T, mu_tilda)
            p.append(p_t_a)

        a_star = np.argmax(p)
        return a_star

    def updateParameters(self, X, action, reward):
        self.B[action]  += np.outer(X, X)
        self.f[action] += reward * X
        self.mu_hat[action] = np.dot(np.linalg.inv(self.B[action]), self.f[action])

class ContextualLinearSVMSLPolicy(Policy):
    def __init__(self,data):
        self.data_prepocessor = DataPipeline()
        self.X_train, self.X_val, self.y_train, self.y_val = data
        self.LinearModel = LinearSVC()

        self.LinearModel.fit(self.X_train,self.y_train)

    def choose(self, X):
        prediction = self.LinearModel.predict(X.reshape(1,X.shape[0]))
        return prediction

    def updateParameters(self, X, action, reward):
        pass

class ContextualSVMSLPolicy(Policy):
    def __init__(self, data):
        self.data_prepocessor = DataPipeline()
        self.X_train, self.X_val, self.y_train, self.y_val = data
        self.lr_model = LogisticRegression(random_state=1,multi_class='multinomial',solver='newton-cg',
                                          verbose=0)

        self.lr_model.fit(self.X_train, self.y_train)

    def choose(self, X):
        prediction = self.lr_model.predict(X.reshape(1,X.shape[0]))

        param_grid = {
            'C': [1, 2, 3],
            'degree': [3, 4, 5, 6, 7 , 8],
        }

        SVM_model = SVC(gamma='auto')

        SVM_Tuned = GridSearchCV(estimator=SVM_model, param_grid=param_grid, cv=StratifiedKFold(3))
        SVM_Tuned.fit(self.X_train, self.y_train)

        self.SVM_model = SVM_Tuned

    def choose(self, X):
        prediction = self.SVM_model.predict(X.reshape(1,X.shape[0]))
        return int(prediction)

    def updateParameters(self, X, action, reward):
        pass