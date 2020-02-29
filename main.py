import numpy as np
import pandas as pd
from sklearn.utils import shuffle
from datapreprocessor import DataPreProcessor

class WarfarinDosageRecommendation(object):

    def __init__(self):
        self.data_prepocessor = DataPreProcessor()
        self.X_train, self.X_val, self.y_train, self.y_val = self.data_prepocessor.loadAndPrepData()

        self.train_size = self.X_train.shape[0]
        self.test_size = self.X_val.shape[0]
        self.num_features = self.train_size.shape[1]



if __name__ == '__main__':
    warfarin = WarfarinDosageRecommendation()