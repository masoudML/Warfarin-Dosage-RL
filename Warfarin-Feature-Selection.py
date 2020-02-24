import pandas as pd
import numpy as np

from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
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

import lightgbm as lgb

### Plot predictions distribution vs ground truth flag
plot_pred_vs_GT = False

### data prep, handling data and cleaning
### explore target (Y) and feature analysis (contintuous and categorical features)
def dataInspectionAndExploration(data, name='train', plotFlag=False):

    print(name+ " data description: ")
    print(data.describe())
    print(data.isnull().sum())

    data = data[data['Therapeutic Dose of Warfarin'].notnull()][['Gender','Race','Age','Ethnicity','Weight (kg)','Height (cm)', 'Therapeutic Dose of Warfarin']]

    if(plotFlag):

        sns.set(style="darkgrid", font_scale=0.55)

        sns.pairplot(data.sample(n=250))
        plt.savefig(name+'pairPlot.png')
        plt.clf()

        targetPlot = sns.distplot(data['Therapeutic Dose of Warfarin'])
        targetPlot.set_title('Therapeutic Dose of Warfarin', fontsize=12)
        plt.savefig(name+'targetPlot.png')
        plt.clf()

        targetBoxPlot = sns.boxplot(data['Therapeutic Dose of Warfarin'])
        targetBoxPlot.set_title('Therapeutic Dose of Warfarin', fontsize=12)
        plt.savefig(name + 'targetBoxPlot.png')
        plt.clf()

        AgePlot = sns.distplot(data['Age'])
        AgePlot.set_title('Age', fontsize=12)
        plt.savefig(name+'AgePlot.png')
        plt.clf()

        weightPlot = sns.boxplot(data['Weight (kg)'])
        weightPlot.set_title('Weight', fontsize=12)
        plt.savefig(name + 'WeightPlot.png')
        plt.clf()

        heightPlot = sns.boxplot(data['Height (cm)'])
        heightPlot.set_title('Height', fontsize=12)
        plt.savefig(name + 'HeightPlot.png')
        plt.clf()


    aggregations = {
        'Therapeutic Dose of Warfarin': {
            'frequency': 'count',
            'min': 'min',
            'mean': 'mean',
            'max': 'max'
        }
    }

    dosage_per_race_stats = data.groupby('Race').agg(aggregations)
    dosage_per_race_stats.to_csv('dosage_per_race_stats.csv', sep=',')

    return data


### one hot encoding for categorical variables
def encodeCategoricalVariables(data, type='train', save=False):

    data['Gender'] = data['Gender'].astype(str)
    data['Age'] = data['Age'].astype(str)
    data = data.apply(LabelEncoder().fit_transform)

    if(save):
        data.to_csv('preprocesssed_data.csv')

    if(type=='train'):
        Y = data['Therapeutic Dose of Warfarin']
        X = data.loc[:, data.columns != 'Therapeutic Dose of Warfarin']
        return X,Y

    return data

### Univariate Linear Models
def runUnivariateTests(X_train, X_val, y_train, y_val, print_results=False):

    ### start with Univariate model analysis
    ## using univariate linear regression model - check the predictive signficance of individual predictor
    ### RMSE, R^2, AIC
    univariate_df = pd.DataFrame(columns=['Model', 'RMSE', 'R^2', 'AIC'])
    for col in X_train.columns:
        X = sm.add_constant(X_train[col])
        model = sm.OLS(y_train, X)
        ols_res = model.fit()
        if (print_results):
            print(ols_res.summary())
        X_V = sm.add_constant(X_val[col])
        pred = ols_res.predict(X_V)
        rmse =  np.sqrt(mean_squared_error(y_val, pred))
        r2 =  ols_res.rsquared
        univariate_df = univariate_df.append({'Model': col,'RMSE': rmse, 'R^2':r2, 'AIC':ols_res.aic}, ignore_index=True)

    univariate_df.to_csv('univariate_models.csv', index=False)

def buildBaselineModel(X_train, X_val, y_train, y_val, show_results=False):

    X = sm.add_constant(X_train)
    model = sm.OLS(y_train, X)
    ols_res = model.fit()
    X_V = sm.add_constant(X_val)
    pred = ols_res.predict(X_V)
    rmse = np.sqrt(mean_squared_error(y_val, pred))
    r2 = ols_res.rsquared

    if (show_results):
        print(ols_res.summary())
        print('baseline rmse :', rmse)
        sns.set(style="darkgrid", font_scale=0.55)
        targetPlot = sns.distplot(y_val, label='Ground Truth', color='blue')
        predPlot = sns.distplot(pred , label='predictions',color='red')
        plt.legend(['Ground Truth', 'predictions'])
        plt.savefig('predvaltargetPlot.pdf')

    model = LinearRegression(normalize=True)
    model.fit(X_train, y_train)
    return model

def oneHotEncoding(X):
    columnsToEncode = X.select_dtypes(include=[object]).columns
    return pd.get_dummies(X, columns=columnsToEncode)


### Data exploration and Preprocessing
def dataLoadingandExplore():
    traindata = pd.read_csv('data/warfarin.csv')

    ## data prep, inspection, cleaining, exploration
    traindata = dataInspectionAndExploration(traindata)

    return traindata

def rank_to_dict(ranks, names, order=1):
    minmax = MinMaxScaler()
    ranks = minmax.fit_transform(order*np.array([ranks]).T).T[0]
    ranks = map(lambda x: round(x, 2), ranks)
    return dict(zip(names, ranks ))


def performFeatureAnalysisUsingRFE(X_train, X_val, y_train, y_val):
    ## run univariate tests to examine each feature individually
    ## to determine the strength of the relationship of the feature with the response variable.
    runUnivariateTests(X_train, X_val, y_train, y_val)

    ### Using LR model - Perform Recursive Feature Elimination to select best set of features
    model = LinearRegression(normalize=True)
    model.fit(X_train,y_train)

    rfecv = RFECV(estimator=model, step=1, cv=StratifiedKFold(3), scoring='neg_mean_squared_error', verbose=0)
    rfecv.fit(X_train, y_train)

    print('RFE Support: ' + str(rfecv.support_))
    print('RFE Rankings: ' + str(rfecv.ranking_))
    ### number of selected features
    print('RFE Rankings: ' + str(rfecv.n_features_))



def performFeatureAnalysisUsingRandomForest(X_train, y_train, plot_rankings=False):
    #### Feature Importance Rankings using Random Forest
    RF_model = RandomForestRegressor(n_estimators=50,
                                     random_state=1,
                                     min_samples_leaf=8,
                                     max_depth=6,
                                     verbose=0)
    RF_model.fit(X_train, y_train)

    ranks = rank_to_dict(RF_model.feature_importances_, X_train.columns)
    ranks = dict(zip(X_train.columns, RF_model.feature_importances_))

    ranks_df = pd.DataFrame(list(ranks.items()), columns=['Feature', 'Ranking'])
    ranks_df = ranks_df.sort_values('Ranking', ascending=False)

    print(ranks_df)

    if (plot_rankings):
        rc = {'axes.labelsize': 14, 'font.size': 14, 'legend.fontsize': 14, 'axes.titlesize': 14}
        sns.set(rc=rc)
        sns.catplot(x="Ranking", y="Feature", data=ranks_df, kind="bar",
                       height=12, aspect=1.9, palette='coolwarm')
        plt.savefig('RFFeatureRankingPlot.png')
        plt.clf()


def main():

    ###################### Data Preprocessing ######################
    ### load, clean and prepare the train data
    traindata = dataLoadingandExplore()
    ## transform Categorical Variables to label encoding - (label encoding is better for Tree Based Regression)
    X, Y, = encodeCategoricalVariables(traindata)
    ### one hot encoding for Multiple Linear Regression and Regularized Mulitple LR
    #X_onehot = oneHotEncoding(traindata.loc[:, traindata.columns != 'Therapeutic Dose of Warfarin'])
    ### split train data into train and hold out validation sets
    X_train, X_val, y_train, y_val = train_test_split(X, Y, test_size=0.1, random_state=1)

    ######### run feature Analysis and Engineering and return best model ######
    performFeatureAnalysisUsingRFE(X_train, X_val, y_train, y_val)
    performFeatureAnalysisUsingRandomForest(X, Y)

    ###################################################################
    #### Regression Models : Linear Regression, #######################
    #### Lasso, Ridge and Random Forest: with Hyper-Parameter Tuning ##

    ######## LR with all features (and categories) #######
    LRmodel = LinearRegression(normalize=True)
    LRmodel.fit(X_train, y_train)
    lr_pred = LRmodel.predict(X_val)
    LR_rmse = np.sqrt(mean_squared_error(y_val, lr_pred))
    LR_r2 = r2_score(y_val, lr_pred)

    print('Linear Regression model RMSE = %f and R^2 = %f' % (LR_rmse, LR_r2))

    ##### regularization methods : Lasso and Ridge #####
    ## Lasso selects strong set of features and force the rest to zero (sparse) ##
    lasso = LassoCV(alphas=[0.0001, 0.0003, 0.0005, 0.001, 0.003, 0.005, 0.01, 0.03, 0.05, 0.1, 0.3, 0.5, 1], cv=StratifiedKFold(3))
    lasso.fit(X_train, y_train)
    lasso_pred = lasso.predict(X_val)

    lasso_rmse = np.sqrt(mean_squared_error(y_val, lasso_pred))
    lasso_r2 = r2_score(y_val, lasso_pred)
    print('Lasso Grid Seasrch Selected Alpha : ', str(lasso.alpha_))
    print('Lasso Regression model RMSE = %f and R^2 = %f' % (lasso_rmse, lasso_r2))

    ## Ridge - L2 Regularization - diffuses the weights to minimize overfitting ###
    ridge = RidgeCV(alphas=[0.01, 0.05, 0.1, 0.5, 1, 1.5, 2, 2.5, 3], cv=StratifiedKFold(3))
    ridge.fit(X_train, y_train)
    ridge_pred = ridge.predict(X_val)
    ridge_rmse = np.sqrt(mean_squared_error(y_val, ridge_pred))
    ridge_r2 = r2_score(y_val, ridge_pred)

    print('Ridge Grid Seasrch Selected Alpha : ', str(ridge.alpha_))
    print('Ridge Regression model RMSE = %f and R^2 = %f' % (ridge_rmse, ridge_r2))

    ## Random Forest and Grid Search for HyperParameter Tuning ######
    #### use label encoding (not hot encoding)
    X_train, X_val, y_train, y_val = train_test_split(X, Y, test_size=0.1, random_state=1)
    param_grid = {
        'n_estimators': [50, 100, 150],
        'max_depth' : [5,6,7],
    }

    RF_model = RandomForestRegressor(random_state=1,
                                      verbose=0)

    RF_Tuned = GridSearchCV(estimator=RF_model, param_grid=param_grid, cv= StratifiedKFold(3))
    RF_Tuned.fit(X_train, y_train)
    rf_pred = RF_Tuned.predict(X_val)
    rf_rmse = np.sqrt(mean_squared_error(y_val, rf_pred))
    rf_r2 = r2_score(y_val, rf_pred)
    print('Random Forest Grid Best Params : ', RF_Tuned.best_params_)
    print('Random Forest Regression model RMSE = %f and R^2 = %f' % (rf_rmse, rf_r2))

    #### Ensemble Hold-out Validation Predictions and plot results distribution vs Ground Truth Distribution ####
    ### Random Forest is excluded. Worst RMSE and R^2
    ensemble_pred = np.mean([lr_pred,lasso_pred,ridge_pred], axis=0)
    ensemble_rmse = np.sqrt(mean_squared_error(y_val, ensemble_pred))
    ensemble_r2 = r2_score(y_val, ridge_pred)
    print('Ensemble RMSE = %f and R^2 = %f' % (ensemble_rmse, ensemble_r2))

    if (plot_pred_vs_GT):
        sns.set(style="darkgrid", font_scale=0.55)
        targetPlot = sns.distplot(y_val, label='Ground Truth', color='blue')
        predPlot = sns.distplot(ensemble_pred, label='predictions', color='red')
        plt.legend(['Ground Truth', 'predictions'])
        plt.savefig('ensemblePredValTargetPlot.png')

    print("Done")


if __name__ == '__main__':
    main()