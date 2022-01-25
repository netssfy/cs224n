import numpy as np
import pandas as pd

def load_raw_data():
    #return np.fromregex('./data/raw.csv', r'"(\\w+)","?(\\w+)"?', np.object)
    return pd.read_csv('./data/raw.csv', quotechar='"', header=None).to_numpy()

def split_data(rawData):
    num = rawData.shape[0]
    train_num = int(0.7 * num)
    test_num = int(0.2 * num)
    return [
        (rawData[:train_num, 0], rawData[:train_num, 1]),
        (rawData[train_num:train_num + test_num, 0], rawData[train_num:train_num + test_num, 1]),
        (rawData[train_num + test_num:, 0], rawData[train_num + test_num:, 1]),
    ]
