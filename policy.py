
from data_pipeline import DataPipeline

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

from sklearn.svm import SVC, LinearSVC


class Policy(object):

    def __init__(self):
        self.regret = []


    def choose(self, X):

        raise NotImplementedError

    def updateParameters(self, X, action, reward):

        raise NotImplementedError



class RandomForestSLPolicy(Policy):

    def __init__(self, data):
        super().__init__()
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
        super().__init__()
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
        super().__init__()
        self.alpha = 1
        self.delta = 0.05
        self.num_actions = num_actions
        self.features_size = features_size
        self.theta = np.zeros((self.num_actions, self.features_size))
        self.A = [np.identity(self.features_size) for _ in range(self.num_actions)]
        self.b = [np.zeros((self.features_size, 1)) for _ in range(self.num_actions)]
        self.time_step = 0
        self.last_p = None
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
        self.regret.append(np.mean(p[a_star] - p))
        self.last_p = p
        return a_star

    def updateParameters(self, X, action, reward):
        self.A[action] += np.outer(X, X)
        self.b[action] += reward * X.reshape(X.shape[0],1)


class ThompsonSamplingContextualBanditPolicy(Policy):
    def __init__(self, features_size, num_actions=3):
        super().__init__()
        ### hyper-parameters
        #### from the paper: v = R . sqrt((24/eps) d log(1/delta))
        self.R = 0.01
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
        #self.v = self.R * np.sqrt(9 * np.max(np.log(1.1), np.log(1/(self.time_step*self.delta)))) ### d feature size can be removed, R is the range but that would be in extract
        #self.v = self.R * np.sqrt(9 * self.features_size * np.log(self.time_step / self.delta))
        self.v = self.R * np.sqrt(9 * self.features_size * max(np.log(1.001),np.log(1/(self.time_step * self.delta))))

        p = []
        for action in range(self.num_actions):
            mu_tilda = np.random.multivariate_normal(mean=self.mu_hat[action].reshape(self.features_size),
                                                     cov=((self.v ** 2) * np.linalg.inv(self.B[action])))
            p_t_a = np.dot(X.T, mu_tilda)
            p.append(p_t_a)

        a_star = np.argmax(p)
        self.regret.append(np.mean(p[a_star] - p))
        self.last_p = p
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


class FixedBaseline(Policy):
    def __init__(self):
        pass

    def choose(self, X):
        return 1

    def updateParameters(self, X, action, reward):
        pass

class ClinicalBaseline(object):
    def __init__(self):
       pass


    def train(self, seed):
        def get_avg_reward(pred,gt):
            reward=0
            total=0
            cumm_r = []
            curr_e = []
            error = 0
            for p,g in zip(pred,gt):
                #print(p,g)
                if p!=g:
                    reward-=1
                    error+=1
                total+=1
                cumm_r.append(reward)
                curr_e.append(error)
            print(reward,total)
            return cumm_r,curr_e,reward/total

        data = pd.read_csv('data/warfarin.csv')
        data = data.dropna(subset=['Therapeutic Dose of Warfarin'])
        Y = data['Therapeutic Dose of Warfarin']
        X = data.drop(columns='Therapeutic Dose of Warfarin')
        X_train, X_val, y_train, y_val = train_test_split(X, Y, test_size=0.01, random_state=seed)

        fixed = 'medium'
        baseline_train = pd.DataFrame(np.asarray([fixed] * len(y_train)), columns=['Dose'])
        baseline_val=pd.DataFrame(np.asarray([fixed]*len(y_val)),columns=['Dose'])

        binned_train = []
        for item in y_train:
            if item < 21:
                binned_train.append('low')
            elif item > 49:
                binned_train.append('high')
            else:
                binned_train.append('medium')

        binned_val = []
        for item in y_val:
            if item < 21:
                binned_val.append('low')
            elif item > 49:
                binned_val.append('high')
            else:
                binned_val.append('medium')

        print('Fixed Baseline Train: ', get_avg_reward(binned_train,baseline_train['Dose']))
        print('Fixed Baseline Val: ', get_avg_reward(binned_val, baseline_val['Dose']))

        ##VAL
        # Imputed nan with mean age: 6
        age_val = []
        for item in X_val['Age']:
            try:
                if '+' in item:
                    age_val.append(9)
                else:
                    n, _ = item.split(' - ')
                    age_val.append(int(n) // 10)
            except:
                age_val.append(6)

        # Impute with mean 168.0620350798291
        height_val = []
        for item in X_val['Height (cm)']:
            if not np.isnan(item):
                height_val.append(item)
            else:
                height_val.append(168.0620350798291)

        # Impute with mean
        weight_val = []
        for item in X_val['Weight (kg)']:
            if not np.isnan(item):
                weight_val.append(item)
            else:
                weight_val.append(77.84642313546424)

        # Asian race, Black or African American, Missing or Mixed race
        a_race_val, b_race_val, m_race_val = [], [], []
        for item in X_val['Race']:
            a_flag, b_flag, m_flag = 0, 0, 0
            if item == 'Asian':
                a_flag = 1
            elif item == 'Black or African American':
                b_flag = 1
            elif item == 'Unknown':
                m_flag = 1
            a_race_val.append(a_flag)
            b_race_val.append(b_flag)
            m_race_val.append(m_flag)

        # Enzyme inducer status = 1 if patient taking carbamazepine, phenytoin, rifampin, or rifampicin, otherwise zero
        # Imputing nan as 0
        enzyme_val = []
        for c, p, r in zip(X_val['Carbamazepine (Tegretol)'], X_val['Phenytoin (Dilantin)'],
                        X_val['Rifampin or Rifampicin']):
            if not (np.isnan(c) or np.isnan(p) or np.isnan(r)) and (c or p or r):
                enzyme_val.append(1)
            else:
                enzyme_val.append(0)

        # Impute with mean
        amioradone_val = []
        for a in X_val['Amiodarone (Cordarone)']:
            if not np.isnan(a):
                amioradone_val.append(a)
            else:
                amioradone_val.append(0.0653416149068323)

        ##TRAIN
        # Imputed nan with mean age: 6
        age_train = []
        for item in X_train['Age']:
            try:
                if '+' in item:
                    age_train.append(9)
                else:
                    n, _ = item.split(' - ')
                    age_train.append(int(n) // 10)
            except:
                age_train.append(6)

        # Impute with mean 168.0620350798291
        height_train = []
        for item in X_train['Height (cm)']:
            if not np.isnan(item):
                height_train.append(item)
            else:
                height_train.append(168.0620350798291)

        # Impute with mean
        weight_train = []
        for item in X_train['Weight (kg)']:
            if not np.isnan(item):
                weight_train.append(item)
            else:
                weight_train.append(77.84642313546424)

        # Asian race, Black or African American, Missing or Mixed race
        a_race_train, b_race_train, m_race_train = [], [], []
        for item in X_train['Race']:
            a_flag, b_flag, m_flag = 0, 0, 0
            if item == 'Asian':
                a_flag = 1
            elif item == 'Black or African American':
                b_flag = 1
            elif item == 'Unknown':
                m_flag = 1
            a_race_train.append(a_flag)
            b_race_train.append(b_flag)
            m_race_train.append(m_flag)

        # Enzyme inducer status = 1 if patient taking carbamazepine, phenytoin, rifampin, or rifampicin, otherwise zero
        # Imputing nan as 0
        enzyme_train = []
        for c, p, r in zip(X_train['Carbamazepine (Tegretol)'], X_train['Phenytoin (Dilantin)'],
                        X_train['Rifampin or Rifampicin']):
            if not (np.isnan(c) or np.isnan(p) or np.isnan(r)) and (c or p or r):
                enzyme_train.append(1)
            else:
                enzyme_train.append(0)

        # Impute with mean
        amioradone_train = []
        for a in X_train['Amiodarone (Cordarone)']:
            if not np.isnan(a):
                amioradone_train.append(a)
            else:
                amioradone_train.append(0.0653416149068323)



        clinical_baseline_train = []  # dose/week
        for a, h, w, ra, rb, rm, e, am in zip(age_train, height_train, weight_train, a_race_train, b_race_train, m_race_train, enzyme_train, amioradone_train):
            sqrted_dose_weekly = 4.0367 - 0.2546 * a + 0.0118 * h + 0.0134 * w - 0.6752 * ra + 0.4060 * rb + 0.0443 * rm + 1.2799 * e - 0.5695 * am
            dose_weekly = (sqrted_dose_weekly ** 2)
            clinical_baseline_train.append(dose_weekly)

        clinical_baseline_val = []  # dose/week
        for a, h, w, ra, rb, rm, e, am in zip(age_val, height_val, weight_val, a_race_val, b_race_val,
                                            m_race_val, enzyme_val, amioradone_val):
            sqrted_dose_weekly = 4.0367 - 0.2546 * a + 0.0118 * h + 0.0134 * w - 0.6752 * ra + 0.4060 * rb + 0.0443 * rm + 1.2799 * e - 0.5695 * am
            dose_weekly = (sqrted_dose_weekly ** 2)
            clinical_baseline_val.append(dose_weekly)

        clinical_baseline_train_binned = []
        y_train_binned = []
        for cb, yt in zip(clinical_baseline_train, y_train):
            if cb < 21:
                clinical_baseline_train_binned.append(0)
            elif cb > 49:
                clinical_baseline_train_binned.append(2)
            else:
                clinical_baseline_train_binned.append(1)
            if yt < 21:
                y_train_binned.append(0)
            elif yt > 49:
                y_train_binned.append(2)
            else:
                y_train_binned.append(1)

        cumm_r, curr_e, reward = get_avg_reward(clinical_baseline_train_binned, y_train_binned)
        
        print('Clinical Baseline Train: ', reward)

        clinical_baseline_val_binned = []
        y_val_binned = []
        for cb, yt in zip(clinical_baseline_val, y_val):
            if cb < 21:
                clinical_baseline_val_binned.append('low')
            elif cb > 49:
                clinical_baseline_val_binned.append('high')
            else:
                clinical_baseline_val_binned.append('medium')
            if yt < 21:
                y_val_binned.append('low')
            elif yt > 49:
                y_val_binned.append('high')
            else:
                y_val_binned.append('medium')

        #cumm_r, reward = get_avg_reward(clinical_baseline_val_binned, y_val_binned)

        print('Clinical Baseline Val: ', reward)
        return y_train_binned, cumm_r, clinical_baseline_train_binned, curr_e

    def updateParameters(self, X, action, reward):
        pass