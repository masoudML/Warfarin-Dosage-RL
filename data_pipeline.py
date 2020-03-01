import pandas as pd
import numpy as np

class DataPipeline(object):
    def __init__(self):
        #No Init needed
        pass
    def null_label_cleaner(self,data):
        """
        data is a dataframe
        """
        data=data.dropna(subset = ['Therapeutic Dose of Warfarin'])
        #Remove Comorbidities since it's encoded as BERT features in our dataframe
        data = data.drop("Comorbidities", axis=1)
        return data

    def unnamed_cleaner(self,data):
        """
        Unnamed and Subject ID are not important
        """
        data=data.drop(columns=['Unnamed: 0', 'Unnamed: 0.1', 'Unnamed: 63','Unnamed: 64','Unnamed: 65'])
        data=data.drop(columns=['PharmGKB Subject ID'])
        return data

    def more_than_half_na_dropper(self,data):
        """
        Drop more than half NA
        """
        cols=data.columns.values
        for col in cols:
            print(col, data[col].isna().sum())
        cols=data.columns.values
        thresh=len(data)//2
        for col in cols:
            if data[col].isna().sum()>thresh:
                data=data.drop(columns=[col])
        return data


    def impute_data(self,data):
        #impute data
        #TODO Rewa to add more imputation methods here
        print('Start of impute: ',len(data))
        
        cols=data.columns.values
        for col in cols:
            if data[col].dtype=='float64':
                data[col].fillna(data[col].mean(),inplace=True)
            else:
                data[col].fillna(data[col].mode().iloc[0],inplace=True)

        #Rewa suggested we drop medications
        data=data.drop(columns=['Medications'])
        
        #Imputation for 
        #data['Indication for Warfarin Treatment']
        #Possible values: ['7', '3', '8', '1', '2', '4', '5', '6', '4; 8', '3; 8', '3; 4',
        #    '3; 4; 8', '1; 6', '7; 8', '3; 6', '3; 6; 8', '6; 8', '4; 7',
        #    '3; 4; 6; 8', '1; 3; 8', '3; 4; 6', '3; 4; 7; 8', '4; 6', '2; 8',
        #    '3; 4; 6; 7; 8', '3; 4; 7', '4; 7; 8', '1; 2', '1; 8', '4; 6; 8',
        #    '1; 2; 8', '2; 3; 8', '3; 6; 7', '3; 7', '3; 7; 8', '1;2', '4;6',
        #    '5; 8', '1; 3', '2; 3', '4; 5', '3; 5', '1; 2; 3', '5; 6',
        #    '1; 3; 4; 8', '1; 2; 5; 8', '3.0', '2.0', '4.0', '8.0', '1.0',
        #    '6.0', '1 or 2', '4; 3', '3; 2', '6; 5', '1,2', '2; 6']
        #Method - create vector from 0 to 8
        list_indication = []
        for row in data['Indication for Warfarin Treatment'].astype(str):
            print(row)
            row_np = np.zeros(9)
            print(row)
            for i in range(0,9):
                if row.find(str(i))>-1:
                    row_np[i] = 1
            #print(row_np)
            list_indication.append(row_np)
        #Drop it
        data=data.drop(columns=['Indication for Warfarin Treatment'])
        
        list_np = np.asarray(list_indication)
        #print('sizeof list_np: ', list_np.shape)
        df_indication = pd.DataFrame(data=list_np, columns=[ "indi"+str(x) for x in range(0,9)])
        #print('len df_ind: ', len(df_indication))
        #print('df_indication',df_indication)
        data.reset_index(drop=True, inplace=True)
        df_indication.reset_index(drop=True, inplace=True)
        data = pd.concat([data, df_indication], axis=1, sort=False)
        #print('End of impute: ',len(data))
        #exit()
        return data


    def convert_labels_to_catg(self,data):
        #low: less than 21 mg/week
        # medium: 21-49 mg/week
        # high: more than 49 mg/week
        print(data)
        data.loc[(data['Therapeutic Dose of Warfarin'] < 21),'Therapeutic Dose of Warfarin']=0
        data.loc[((data['Therapeutic Dose of Warfarin'] >= 21) & (data['Therapeutic Dose of Warfarin'] < 49)),'Therapeutic Dose of Warfarin']=1
        data.loc[(data['Therapeutic Dose of Warfarin'] >= 49),'Therapeutic Dose of Warfarin']=2
        return data
        
    def convert_to_one_hot(self,data):
        cols=data.columns.values
        new_df = pd.DataFrame()
        for col in cols:
            if data[col].dtype=='float64':
                new_df = pd.concat([new_df, data[col]], axis=1)
            else:
                data[col] = pd.Categorical(data[col])
                dfDummies = pd.get_dummies(data[col], prefix = 'catg')
                print('**********')
                print('Col name:',col)
                print('# values are:',len(dfDummies.columns))
                print('**********')
                new_df = pd.concat([new_df, dfDummies], axis=1)
        print('Converted to 1 Hot')
        #print(data)
        return new_df


    def train_test_split(self,data):
        pass

def main():
    data = pd.read_csv('warfarin_pca_bert.csv')
    dp = DataPipeline()
    data = dp.null_label_cleaner(data)
    data = dp.unnamed_cleaner(data)
    data = dp.more_than_half_na_dropper(data)
    data = dp.impute_data(data)
    data = dp.convert_labels_to_catg(data)
    data = dp.convert_to_one_hot(data)

    print(data)




if __name__ == "__main__":
    main()