import numpy as np
import pandas as pd
from sklearn.utils import shuffle
from data_pipeline import DataPipeline
from policy import RandomForestSLPolicy, ContextualLinearUCBPolicy, LogisticRegressionSLPolicy, ThompsonSamplingContextualBanditPolicy, ClinicalBaseline
from sklearn.metrics import precision_recall_fscore_support, classification_report, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import gc
from sklearn.metrics import precision_recall_fscore_support

from data_pipeline import DataPipeline
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split





class WarfarinDosageRecommendation(object):

    def __init__(self, policy,data ):
        self.X_train, self.X_val, self.y_train, self.y_val = data
        self.train_size = self.X_train.shape[0]
        self.test_size = self.X_val.shape[0]
        self.features_size = self.X_train.shape[1]
        self.policy = policy

    def calculateReward(self, action, y):
        return 0 if int(action) == int(y) else -1

    def train(self, X_train, y_train, epochs=1):

        X_train = X_train.values
        y_train = y_train.values

        errors = []
        cum_errors = []

        for epoch in range(epochs):
            predictions = []
            rewards = []
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
    #seeds = [1,12,123,1234, 12345, 1234545, 0, 2, 234, 2345, 23454, 345, 3456, 345656, 456, 45656, 7483, 7590 , 789, 7890 ]
    #seeds = np.random.randint(2 ** 30, size=20)
    seeds = np.random.randint(2 ** 30, size=20)
    data_prepocessor = DataPipeline()#(bert_on=False)
    X_train, X_val, y_train, y_val = data_prepocessor.loadAndPrepData()
    linUCB_regrets = []
    ts_regrets = []
    linUCB_cum_errors = []
    ts_cum_errors = []
    softmax_cum_errors = []
    RF_cum_errors = []
    baseline_cum_errors = []
    fixed_cum_errors = []

    fixed_policy = FixedBaseline(features_size=X_train.shape[1], num_actions=3)
    fixed_warfarin = WarfarinDosageRecommendation(fixed_policy, data=(X_train, X_val, y_train, y_val))


    linUCB_policy = ContextualLinearUCBPolicy(features_size=X_train.shape[1], num_actions=3)
    warfarin = WarfarinDosageRecommendation(linUCB_policy, data=(X_train, X_val, y_train, y_val))

    t_policy = ThompsonSamplingContextualBanditPolicy(features_size=X_train.shape[1], num_actions=3)
    warfarin_t = WarfarinDosageRecommendation(t_policy, data=(X_train, X_val, y_train, y_val))

    softmax_policy = LogisticRegressionSLPolicy(data=(X_train, X_val, y_train, y_val))
    warfarin_soft = WarfarinDosageRecommendation(softmax_policy, data=(X_train, X_val, y_train, y_val))

    rf_policy = RandomForestSLPolicy(data=(X_train, X_val, y_train, y_val))
    warfarin_rf = WarfarinDosageRecommendation(rf_policy, data=(X_train, X_val, y_train, y_val))

    bl_policy = ClinicalBaseline()
    #warfarin_bl = WarfarinDosageRecommendation(bl_policy, data=(X_train, X_val, y_train, y_val))

    linUCB_accuracy, ts_accuracy, softmax_accuracy, rf_accuracy, bl_accuracy, fixed_accuracy = [],[],[],[], [],[]
    linUCB_precision, linUCB_recall, linUCB_fscore = [],[],[]
    ts_precision, ts_recall, ts_fscore = [], [], []
    softmax_precision,softmax_recall, softmax_fscore = [], [], []
    rf_precision,rf_recall, rf_fscore = [], [], []
    bl_precision,bl_recall, bl_fscore = [], [], []
    fixed_precision, fixed_recall, fixed_fscore = [],[],[]

    for i in range(len(seeds)):
        print(' ########## seed {} = {} ###############'.format(i, seeds[i]))
        X_train_shuffled, y_train_shuffled = shuffle(X_train, y_train, random_state=seeds[i])
        #linUCB_policy = ContextualLinearUCBPolicy(features_size=X_train.shape[1], num_actions=3)
        #warfarin = WarfarinDosageRecommendation(linUCB_policy, data=(X_train, X_val, y_train, y_val))
        rewards, predictions, cum_errors = warfarin.train(X_train_shuffled, y_train_shuffled, epochs=1)
        linUCB_cum_errors.append(cum_errors)
        print('########################### LinUCB ########################################')
        print('##### Train #### ')
        accuracy = accuracy_score(y_train_shuffled, predictions)
        linUCB_accuracy.append(accuracy)
        print('accuracy: ' + str(accuracy))
        print(classification_report(y_train_shuffled, predictions))
        precision, recall, fscore, _ = precision_recall_fscore_support(y_train_shuffled, predictions, average='weighted')
        linUCB_precision.append(precision)
        linUCB_recall.append(recall)
        linUCB_fscore.append(fscore)

        print('LinUCB: Avg Reward on the train: {} '.format(np.mean(rewards)))
        linUCB_regrets.append(linUCB_policy.regret)

        #t_policy = ThompsonSamplingContextualBanditPolicy(features_size=X_train.shape[1], num_actions=3)
        #warfarin = WarfarinDosageRecommendation(t_policy, data=(X_train, X_val, y_train, y_val))
        rewards, predictions, cum_errors = warfarin_t.train(X_train_shuffled, y_train_shuffled, epochs=1)
        ts_cum_errors.append(cum_errors)
        print('########################### TS Contextual bandit ########################################')
        print('##### Train #### ')
        accuracy = accuracy_score(y_train_shuffled, predictions)
        ts_accuracy.append(accuracy)
        print('accuracy: ' + str(accuracy))
        print(classification_report(y_train_shuffled, predictions))
        precision, recall, fscore, _ = precision_recall_fscore_support(y_train_shuffled, predictions,
                                                                       average='weighted')
        ts_precision.append(precision)
        ts_recall.append(recall)
        ts_fscore.append(fscore)
        print('TS: Avg Reward on the train: {} '.format(np.mean(rewards)))
        ts_regrets.append(t_policy.regret)

        print('########################### Logistic (Softmax) Regression ########################################')
        print('##### Train #### ')

        rewards, predictions, cum_errors = warfarin_soft.train(X_train_shuffled, y_train_shuffled)
        softmax_cum_errors.append(cum_errors)
        accuracy = accuracy_score(y_train_shuffled, predictions)
        softmax_accuracy.append(accuracy)
        print('accuracy: ' + str(accuracy))
        print(classification_report(y_train_shuffled, predictions))
        precision, recall, fscore, _ = precision_recall_fscore_support(y_train_shuffled, predictions,
                                                                       average='weighted')
        softmax_precision.append(precision)
        softmax_recall.append(recall)
        softmax_fscore.append(fscore)
        print('Softmax: Avg Reward on the train: {} '.format(np.mean(rewards)))

        print('########################### Random Forest ########################################')
        print('##### Train #### ')

        rewards, predictions, cum_errors = warfarin_rf.train(X_train_shuffled, y_train_shuffled)
        RF_cum_errors.append(cum_errors)
        accuracy = accuracy_score(y_train_shuffled, predictions)
        rf_accuracy.append(accuracy)
        print('accuracy: ' + str(accuracy))
        print(classification_report(y_train_shuffled, predictions))
        precision, recall, fscore, _ = precision_recall_fscore_support(y_train_shuffled, predictions,
                                                                       average='weighted')
        rf_precision.append(precision)
        rf_recall.append(recall)
        rf_fscore.append(fscore)
        print('RF: Avg Reward on the train: {} '.format(np.mean(rewards)))



        print('########################### BASELINE ########################################')
        print('##### Train #### ')

        #rewards, predictions, cum_errors = warfarin_bl.train(X_train_shuffled, y_train_shuffled)
        y_train_shuffled, rewards, predictions, cum_errors = bl_policy.train(i)
        baseline_cum_errors.append(cum_errors)
        accuracy = accuracy_score(y_train_shuffled, predictions)
        bl_accuracy.append(accuracy)
        print('accuracy: ' + str(accuracy))
        print(classification_report(y_train_shuffled, predictions))
        precision, recall, fscore, _ = precision_recall_fscore_support(y_train_shuffled, predictions,
                                                                       average='weighted')
        bl_precision.append(precision)
        bl_recall.append(recall)
        bl_fscore.append(fscore)
        print('BASELINE: Avg Reward on the train: {} '.format(np.mean(rewards)))


        print('########################### FIXED ########################################')
        print('##### Train #### ')

        rewards, predictions, cum_errors = fixed_warfarin.train(X_train_shuffled, y_train_shuffled, epochs=1)
        fixed_cum_errors.append(cum_errors)
        accuracy = accuracy_score(y_train_shuffled, predictions)
        fixed_accuracy.append(accuracy)
        print('accuracy: ' + str(accuracy))
        print(classification_report(y_train_shuffled, predictions))
        precision, recall, fscore, _ = precision_recall_fscore_support(y_train_shuffled, predictions, average='weighted')
        fixed_precision.append(precision)
        fixed_recall.append(recall)
        fixed_fscore.append(fscore)

        gc.collect()

    results_table = pd.DataFrame(columns=['Model', 'Accuracy', 'Weighted_Precision', 'Weighted_Recall', 'Weighted_Fscore'])
    results_table = results_table.append(pd.DataFrame({'Model': ['LinUCB','TS','Softmax','RF','BL-Clinical','BL-Fixed'],
                  'Accuracy': [np.mean(linUCB_accuracy),np.mean(ts_accuracy),np.mean(softmax_accuracy),np.mean(rf_accuracy),np.mean(bl_accuracy),np.mean(fixed_accuracy)],
                  'Weighted_Precision': [np.mean(linUCB_precision),np.mean(ts_precision),np.mean(softmax_precision),np.mean(rf_precision),np.mean(bl_precision),np.mean(fixed_precision)],
                  'Weighted_Recall': [np.mean(linUCB_recall),np.mean(ts_recall),np.mean(softmax_recall),np.mean(rf_recall),np.mean(bl_recall),np.mean(fixed_recall)],
                  'Weighted_Fscore': [np.mean(linUCB_fscore), np.mean(ts_fscore),np.mean(softmax_fscore), np.mean(rf_fscore),np.mean(bl_fscore),np.mean(fixed_fscore)]}))


    print('########### Results Table ########## ')
    print(results_table)

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
    ax.set_xlim(-500, None)
    ax.set(xlabel='Time', ylabel='Cumulative Regret')
    ax.legend(loc=0, labels=["LinUCB", "TS-Contextual-Bandit"])
    fig.savefig("regret_all.png")
    fig.clf()

    fixed_cum_errors = np.array(fixed_cum_errors)
    fixed_cum_errors = pd.DataFrame(data=fixed_cum_errors.T, columns=list(range(fixed_cum_errors.shape[0])),
                                 index=list(range(fixed_cum_errors.shape[1])))


    baseline_cum_errors = np.array(baseline_cum_errors)
    baseline_cum_errors = pd.DataFrame(data=baseline_cum_errors.T, columns=list(range(baseline_cum_errors.shape[0])),
                                 index=list(range(baseline_cum_errors.shape[1])))


    linUCB_cum_errors = np.array(linUCB_cum_errors)
    linUCB_cum_Errors = pd.DataFrame(data=linUCB_cum_errors.T, columns=list(range(linUCB_cum_errors.shape[0])),
                                 index=list(range(linUCB_cum_errors.shape[1])))

    ts_cum_errors = np.array(ts_cum_errors)
    ts_cum_Errors = pd.DataFrame(data=ts_cum_errors.T, columns=list(range(ts_cum_errors.shape[0])),
                             index=list(range(ts_cum_errors.shape[1])))

    softmax_cum_errors = np.array(softmax_cum_errors)
    softmax_cum_Errors = pd.DataFrame(data=softmax_cum_errors.T, columns=list(range(softmax_cum_errors.shape[0])),
                                     index=list(range(softmax_cum_errors.shape[1])))

    RF_cum_errors = np.array(RF_cum_errors)
    RF_cum_Errors = pd.DataFrame(data=RF_cum_errors.T, columns=list(range(RF_cum_errors.shape[0])),
                                 index=list(range(RF_cum_errors.shape[1])))

    fig, ax = plt.subplots(figsize=(15, 10))
    sns.tsplot(data=linUCB_cum_Errors.values.T, ci=95, estimator=np.mean, color='m', ax=ax, legend=True).set_title(
        'Algs Cumulative Errors Rate')
    sns.tsplot(data=ts_cum_Errors.values.T, ci=95, estimator=np.mean, color='r', ax=ax, legend=True)
    sns.tsplot(data=softmax_cum_Errors.values.T, ci=95, estimator=np.mean, color='g', ax=ax, legend=True)
    sns.tsplot(data=RF_cum_Errors.values.T, ci=95, estimator=np.mean, color='b', ax=ax, legend=True)
    sns.tsplot(data=baseline_cum_errors.values.T, ci=95, estimator=np.mean, color='y', ax=ax, legend=True)
    sns.tsplot(data=fixed_cum_errors.values.T, ci=95, estimator=np.mean, color='k', ax=ax, legend=True)



    ax.set_xlim(-500, None)
    ax.set(xlabel='Time', ylabel='Cumulative Error Rate')
    ax.legend(loc=0, labels=["LinUCB", "TS-Contextual-Bandit","Softmax Regression", "Random Forest", "Clinical Baseline","Fixed Baseline"])
    fig.savefig("errors_all.png")
    fig.clf()

    ### Test on the validation set - multiple episodes

    X_train, X_val, y_train, y_val = data_prepocessor.loadAndPrepData(test_size=0.1)

    linUCB_policy = ContextualLinearUCBPolicy(features_size=X_train.shape[1], num_actions=3)
    warfarin = WarfarinDosageRecommendation(linUCB_policy, data=(X_train, X_val, y_train, y_val))
    rewards, predictions, cum_errors = warfarin.train(X_train, y_train, epochs=5)
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
    rewards, predictions, cum_errors = warfarin.train(X_train, y_train, epochs=5)
    ts_regret = t_policy.regret
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
    rewards, predictions, cum_errors = warfarin.train(X_train, y_train)
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
    rewards, predictions, cum_errors = warfarin.train(X_train, y_train)
    print('accuracy: ' + str(accuracy_score(y_train, predictions)))
    print(classification_report(y_train, predictions))
    print('RF: Avg Reward on the train: {} '.format(np.mean(rewards)))

    print('##### VAL #### ')
    rewards, predictions = warfarin.eval()
    print('accuracy: ' + str(accuracy_score(y_val, predictions)))
    print(classification_report(y_val, predictions))
    print('RF: Avg Reward on the val: {} '.format(np.mean(rewards)))

