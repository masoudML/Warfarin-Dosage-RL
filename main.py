import numpy as np
import pandas as pd
from sklearn.utils import shuffle
from datapreprocessor import DataPreProcessor
from data_pipeline import DataPipeline
from policy import RandomForestSLPolicy, ContextualLinearUCBPolicy, LogisticRegressionSLPolicy, ThompsonSamplingContextualBanditPolicy
from sklearn.metrics import precision_recall_fscore_support, classification_report, accuracy_score

class WarfarinDosageRecommendation(object):

    def __init__(self, policy,data ):
        self.X_train, self.X_val, self.y_train, self.y_val = data
        self.train_size = self.X_train.shape[0]
        self.test_size = self.X_val.shape[0]
        self.features_size = self.X_train.shape[1]
        self.policy = policy

    def calculateReward(self, action, y):
        return 0 if int(action) == int(y) else -1

    def train(self, epochs=1):

        # TODO: shuffle



        X_train = self.X_train.values
        y_train = self.y_train.values

        #X_train, y_train = shuffle(X_train, y_train, random_state=123)

        for epoch in range(epochs):
            predictions = []
            rewards = []
            for t in range(self.train_size):
                X = X_train[t,:]
                y = y_train[t]
                action = self.policy.choose(X)
                reward = self.calculateReward(action, y)

                predictions.append(action)
                self.policy.updateParameters(X, action, reward)
                rewards.append(reward)

        return (rewards, predictions)

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

        return (rewards, predictions)

if __name__ == '__main__':
    data_prepocessor = DataPipeline()
    X_train, X_val, y_train, y_val = data_prepocessor.loadAndPrepData()
    linUCB_policy = ContextualLinearUCBPolicy(features_size=X_train.shape[1], num_actions=3)
    warfarin = WarfarinDosageRecommendation(linUCB_policy, data=(X_train, X_val, y_train, y_val))
    rewards, predictions = warfarin.train(epochs=5)
    print('########################### LinUCB ########################################')
    print('##### Train #### ')
    print('accuracy: ' + str(accuracy_score(y_train, predictions)))
    print(classification_report(y_train, predictions))

    print('LinUCB: Avg Reward on the train: {} '.format(np.mean(rewards)))

    print('##### VAL #### ')
    rewards, predictions = warfarin.eval()

    print('accuracy: ' + str(accuracy_score(y_val, predictions)))
    print(classification_report(y_val, predictions))
    print('LinUCB: Avg Reward on the val: {} '.format(np.mean(rewards)))

    print('\n')
    print('########################### TS for Contextual Bandits ########################################')
    print('##### Train #### ')
    t_policy = ThompsonSamplingContextualBanditPolicy(features_size=X_train.shape[1], num_actions=3)
    warfarin = WarfarinDosageRecommendation(t_policy, data=(X_train, X_val, y_train, y_val))
    rewards, predictions = warfarin.train(epochs=5)
    print('accuracy: ' + str(accuracy_score(y_train, predictions)))
    print(classification_report(y_train, predictions))
    print('TS for Contextual Bandits: Avg Reward on the train: {} '.format(np.mean(rewards)))

    print('##### VAL #### ')
    rewards, predictions = warfarin.eval()
    print('accuracy: ' + str(accuracy_score(y_val, predictions)))
    print(classification_report(y_val, predictions))
    print('TS for Contextual Bandits: Avg Reward on the val: {} '.format(np.mean(rewards)))

    print('\n')
    print('########################### Logistic (Softmax) Regression ########################################')
    print('##### Train #### ')
    softmax_policy = LogisticRegressionSLPolicy(data=(X_train, X_val, y_train, y_val))
    warfarin = WarfarinDosageRecommendation(softmax_policy, data=(X_train, X_val, y_train, y_val))
    rewards, predictions = warfarin.train()
    print('accuracy: ' + str(accuracy_score(y_train, predictions)))
    print(classification_report(y_train, predictions))
    print('Softmax: Avg Reward on the train: {} '.format(np.mean(rewards)))

    print('##### VAL #### ')
    rewards, predictions = warfarin.eval()
    print('accuracy: ' + str(accuracy_score(y_val, predictions)))
    print(classification_report(y_val, predictions))
    print('Softmax: Avg Reward on the val: {} '.format(np.mean(rewards)))

    print('\n')
    print('########################### Random Forest ########################################')
    print('##### Train #### ')
    rf_policy = RandomForestSLPolicy(data=(X_train, X_val, y_train, y_val))
    warfarin = WarfarinDosageRecommendation(rf_policy, data=(X_train, X_val, y_train, y_val))
    rewards, predictions = warfarin.train()
    print('accuracy: ' + str(accuracy_score(y_train, predictions)))
    print(classification_report(y_train, predictions))
    print('RF: Avg Reward on the train: {} '.format(np.mean(rewards)))

    print('##### VAL #### ')
    rewards, predictions = warfarin.eval()
    print('accuracy: ' + str(accuracy_score(y_val, predictions)))
    print(classification_report(y_val, predictions))
    print('RF: Avg Reward on the val: {} '.format(np.mean(rewards)))

    lin_policy = ContextualLinearSVMSLPolicy(data=(X_train, X_val, y_train, y_val))
    warfarin = WarfarinDosageRecommendation(lin_policy, data=(X_train, X_val, y_train, y_val))
    rewards = warfarin.train()
    print('Lin: Avg Reward on the train: {} '.format(np.mean(rewards)))
    rewards = warfarin.eval()
    print('Lin: Avg Reward on the val: {} '.format(np.mean(rewards)))

    svm_policy = ContextualSVMSLPolicy(data=(X_train, X_val, y_train, y_val))
    warfarin = WarfarinDosageRecommendation(svm_policy, data=(X_train, X_val, y_train, y_val))
    rewards = warfarin.train()
    print('SVM: Avg Reward on the train: {} '.format(np.mean(rewards)))
    rewards = warfarin.eval()
    print('SVM: Avg Reward on the val: {} '.format(np.mean(rewards)))