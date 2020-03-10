from data_pipeline import DataPipeline
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def get_avg_reward(pred,gt):
    reward=0
    total=0
    for p,g in zip(pred,gt):
        #print(p,g)
        if p!=g:
            reward-=1
        total+=1
    print(reward,total)
    return reward/total

if __name__=='__main__':

    data = pd.read_csv('data/warfarin.csv')
    data = data.dropna(subset=['Therapeutic Dose of Warfarin'])
    Y = data['Therapeutic Dose of Warfarin']
    X = data.drop(columns='Therapeutic Dose of Warfarin')
    X_train, X_val, y_train, y_val = train_test_split(X, Y, test_size=0.1, random_state=1)

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
            clinical_baseline_train_binned.append('low')
        elif cb > 49:
            clinical_baseline_train_binned.append('high')
        else:
            clinical_baseline_train_binned.append('medium')
        if yt < 21:
            y_train_binned.append('low')
        elif yt > 49:
            y_train_binned.append('high')
        else:
            y_train_binned.append('medium')

    reward = get_avg_reward(clinical_baseline_train_binned, y_train_binned)
    
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

    reward = get_avg_reward(clinical_baseline_val_binned, y_val_binned)

    print('Clinical Baseline Val: ', reward)