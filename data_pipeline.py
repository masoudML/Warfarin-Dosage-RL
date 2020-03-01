import pandas as pd


class DataPipeline(object):
    def __init__(self):
        #No Init needed
        pass
    def null_label_cleaner(self,data):
        """
        data is a dataframe
        """
        data=data.dropna(subset = ['Therapeutic Dose of Warfarin'])
        return data

    def unnamed_cleaner(self,data):
        """
        Unnamed and Subject ID are not important
        """
        data=data.drop(columns=['Unnamed: 63','Unnamed: 64','Unnamed: 65'])
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
        cols=data.columns.values
        for col in cols:
            if data[col].dtype=='float64':
                data[col].fillna(data[col].mean(),inplace=True)
            else:
                data[col].fillna(data[col].mode().iloc[0],inplace=True)

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
        
def main():
    data = pd.read_csv('warfarin_pca_bert.csv')
    dp = DataPipeline()
    data = dp.null_label_cleaner(data)
    data = dp.unnamed_cleaner(data)
    data = dp.more_than_half_na_dropper(data)
    data = dp.impute_data(data)
    data = dp.convert_labels_to_catg(data)
    print(data)




if __name__ == "__main__":
    main()