import numpy as np
import pandas as pd
from sklearn.utils import shuffle
from data_pipeline import DataPipeline
from policy import RandomForestSLPolicy, ContextualLinearUCBPolicy, LogisticRegressionSLPolicy, ThompsonSamplingContextualBanditPolicy
from sklearn.metrics import precision_recall_fscore_support, classification_report, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

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

        X_train = self.X_train.values
        y_train = self.y_train.values

        for epoch in range(epochs):
            predictions = []
            rewards = []
            errors = []
            cum_errors = []
            for t in range(self.train_size):
                X = X_train[t,:]
                y = y_train[t]
                action = self.policy.choose(X)
                reward = self.calculateReward(action, y)

                predictions.append(float(action))
                self.policy.updateParameters(X, action, reward)
                rewards.append(reward)
                errors.append(int(int(action) != int(y)))
                cum_errors.append(np.sum(errors) / len(errors))

        return (rewards, predictions,cum_errors)

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
    seeds = [1,12,123,1234, 12345, 1234545, 0, 2, 234, 2345, 23454, 345, 3456, 345656, 456, 45656, 7483, 7590 , 789, 7890 ]
    data_prepocessor = DataPipeline()
    X_train, X_val, y_train, y_val = data_prepocessor.loadAndPrepData()
    linUCB_regrets = []
    ts_regrets = []
    linUCB_cum_errors = []
    ts_cum_errors = []
    for i in range(len(seeds)):
        print(' ########## seed {} = {} ###############'.format(i, seeds[i]))
        X_train, y_train = shuffle(X_train, y_train, random_state=seeds[i])
        linUCB_policy = ContextualLinearUCBPolicy(features_size=X_train.shape[1], num_actions=3)
        warfarin = WarfarinDosageRecommendation(linUCB_policy, data=(X_train, X_val, y_train, y_val))
        rewards, predictions, cum_errors = warfarin.train(epochs=1)
        linUCB_cum_errors.append(cum_errors)
        print('########################### LinUCB ########################################')
        print('##### Train #### ')
        print('accuracy: ' + str(accuracy_score(y_train, predictions)))
        print(classification_report(y_train, predictions))
        print('LinUCB: Avg Reward on the train: {} '.format(np.mean(rewards)))
        linUCB_regrets.append(linUCB_policy.regret)

        t_policy = ThompsonSamplingContextualBanditPolicy(features_size=X_train.shape[1], num_actions=3)
        warfarin = WarfarinDosageRecommendation(t_policy, data=(X_train, X_val, y_train, y_val))
        rewards, predictions, cum_errors = warfarin.train(epochs=1)
        ts_cum_errors.append(cum_errors)
        print('########################### TS Contextual bandit ########################################')
        print('##### Train #### ')
        print('accuracy: ' + str(accuracy_score(y_train, predictions)))
        print(classification_report(y_train, predictions))
        print('TS: Avg Reward on the train: {} '.format(np.mean(rewards)))
        ts_regrets.append(t_policy.regret)


    linUCB_regrets = np.array(linUCB_regrets)
    linUCB_Regret = pd.DataFrame( data = np.cumsum(linUCB_regrets, axis=1).T, columns=list(range(linUCB_regrets.shape[0])) ,
        index=list(range(linUCB_regrets.shape[1])))

    ts_regrets = np.array(ts_regrets)
    ts_Regret = pd.DataFrame(data=np.cumsum(ts_regrets, axis=1).T, columns=list(range(ts_regrets.shape[0])),
                                 index=list(range(ts_regrets.shape[1])))

    fig, ax = plt.subplots(figsize=(15, 10))
    sns.tsplot(data=linUCB_Regret.values.T, ci=95, estimator=np.mean, color='m', ax=ax, legend=True).set_title(
        'Algs Cumulative Regret')
    sns.tsplot(data=ts_Regret.values.T, ci=95, estimator=np.mean, color='r', ax=ax, legend=True)
    ax.set_xlim(-100, 6000)
    ax.set(xlabel='Time', ylabel='Cumulative Regret')
    ax.legend(loc=0, labels=["LinUCB", "TS-Contextual-Bandit"])
    fig.savefig("regret.png")
    fig.clf()

    linUCB_cum_errors = np.array(linUCB_cum_errors)
    linUCB_cum_Errors = pd.DataFrame(data=linUCB_cum_errors.T, columns=list(range(linUCB_cum_errors.shape[0])),
                                 index=list(range(linUCB_cum_errors.shape[1])))

    ts_cum_errors = np.array(ts_cum_errors)
    ts_cum_Errors = pd.DataFrame(data=ts_cum_errors.T, columns=list(range(ts_cum_errors.shape[0])),
                             index=list(range(ts_cum_errors.shape[1])))

    fig, ax = plt.subplots(figsize=(15, 10))
    sns.tsplot(data=linUCB_cum_Errors.values.T, ci=95, estimator=np.mean, color='m', ax=ax, legend=True).set_title(
        'Algs Cumulative Errors %')
    sns.tsplot(data=ts_cum_Errors.values.T, ci=95, estimator=np.mean, color='r', ax=ax, legend=True)
    ax.set_xlim(-100, 6000)
    ax.set(xlabel='Time', ylabel='Cumulative Error %')
    ax.legend(loc=0, labels=["LinUCB", "TS-Contextual-Bandit"])
    fig.savefig("errors.png")
    fig.clf()

    seed_num = np.random.random_integers(0,19,1)
    X_train, y_train = shuffle(X_train, y_train, random_state=seeds[seed_num[0]])
    linUCB_policy = ContextualLinearUCBPolicy(features_size=X_train.shape[1], num_actions=3)
    warfarin = WarfarinDosageRecommendation(linUCB_policy, data=(X_train, X_val, y_train, y_val))
    rewards, predictions, cum_errors = warfarin.train(epochs=10)
    linUCB_regret = linUCB_policy.regret
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
    rewards, predictions, cum_errors = warfarin.train(epochs=10)
    ts_regret = t_policy.regret
    print('accuracy: ' + str(accuracy_score(y_train, predictions)))
    print(classification_report(y_train, predictions))
    print('TS for Contextual Bandits: Avg Reward on the train: {} '.format(np.mean(rewards)))

    print('##### VAL #### ')
    rewards, predictions = warfarin.eval()
    print('accuracy: ' + str(accuracy_score(y_val, predictions)))
    print(classification_report(y_val, predictions))
    print('TS for Contextual Bandits: Avg Reward on the val: {} '.format(np.mean(rewards)))

    linUCB_Regret = pd.DataFrame(
        {'LinUCB':np.cumsum(linUCB_regret)},
        index=list(range(len(linUCB_regret))))

    TS_Regret = pd.DataFrame(
        {'TS': np.cumsum(ts_regret)},
        index=list(range(len(linUCB_regret))))

    fig, ax = plt.subplots(figsize=(15, 10))

    linUCB_Regret.plot(label='LinUCB', ax=ax)
    TS_Regret.plot(label='ThompsonSampling', ax=ax)
    plt.legend()
    plt.title(r'Cumulative Regret')
    plt.ylabel('Cumulative Regret')
    fig.savefig("regret.png")
    fig.clf()



    print('\n')
    print('########################### Logistic (Softmax) Regression ########################################')
    print('##### Train #### ')
    softmax_policy = LogisticRegressionSLPolicy(data=(X_train, X_val, y_train, y_val))
    warfarin = WarfarinDosageRecommendation(softmax_policy, data=(X_train, X_val, y_train, y_val))
    rewards, predictions, cum_errors = warfarin.train()
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
    rewards, predictions, cum_errors = warfarin.train()
    print('accuracy: ' + str(accuracy_score(y_train, predictions)))
    print(classification_report(y_train, predictions))
    print('RF: Avg Reward on the train: {} '.format(np.mean(rewards)))

    print('##### VAL #### ')
    rewards, predictions = warfarin.eval()
    print('accuracy: ' + str(accuracy_score(y_val, predictions)))
    print(classification_report(y_val, predictions))
    print('RF: Avg Reward on the val: {} '.format(np.mean(rewards)))

