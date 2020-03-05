from data_pipeline import DataPipeline
import pandas as pd
import numpy as np

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

    data_prepocessor = DataPipeline()
    X_train, X_val, y_train, y_val = data_prepocessor.loadAndPrepData()

    fixed=1.0
    baseline_train=pd.DataFrame(np.asarray([fixed]*len(y_train)),columns=['Dose'])
    baseline_val=pd.DataFrame(np.asarray([fixed]*len(y_val)),columns=['Dose'])
    print(type(y_train))
    print('Baseline Train: ', get_avg_reward(y_train,baseline_train['Dose']))
    print('Baseline Val: ', get_avg_reward(y_val, baseline_val['Dose']))