import numpy as np
import pandas as pd
from sklearn.utils import shuffle
from datapreprocessor import DataPreProcessor
from data_pipeline import DataPipeline
from policy import ContextualRandomForestSLPolicy, LinUCBPolicy

class WarfarinDosageRecommendation(object):

    def __init__(self, policy,data ):
        self.X_train, self.X_val, self.y_train, self.y_val = data
        self.train_size = self.X_train.shape[0]
        self.test_size = self.X_val.shape[0]
        self.features_size = self.X_train.shape[1]
        self.policy = policy

    def calculateReward(self, action, y):
        return 0 if int(action) == int(y) else -1

    def train(self):

        # TODO: shuffle

        rewards = []
        predictions = []
        X_train = self.X_train.values
        y_train = self.y_train.values
        for t in range(self.train_size):
            X = X_train[t,:]
            y = y_train[t]
            action = self.policy.choose(X)
            reward = self.calculateReward(action, y)

            predictions.append(action)
            self.policy.updateParameters(X, action, reward)
            rewards.append(reward)

        return rewards

    def eval(self):
        rewards = []
        predictions = []
        X_val = self.X_val.values
        y_val = self.y_val.values
        for t in range(self.test_size):
            X = X_val[t, :]
            y = y_val[t]
            action = self.policy.choose(X)
            reward = self.calculateReward(action, y)

            predictions.append(action)
            #self.policy.updateParameters(X, action, reward)
            rewards.append(reward)

        return rewards

if __name__ == '__main__':
    data_prepocessor = DataPipeline()
    X_train, X_val, y_train, y_val = data_prepocessor.loadAndPrepData()
    linUCB_policy = LinUCBPolicy(features_size=X_train.shape[1], num_actions=3)
    warfarin = WarfarinDosageRecommendation(linUCB_policy, data=(X_train, X_val, y_train, y_val))
    rewards = warfarin.train()
    rewards = warfarin.eval()

    print(np.mean(rewards))

    rf_policy = ContextualRandomForestSLPolicy(data=(X_train, X_val, y_train, y_val))
    warfarin = WarfarinDosageRecommendation(rf_policy, data=(X_train, X_val, y_train, y_val))
    rewards = warfarin.train()
    rewards = warfarin.eval()
    print(np.mean(rewards))


