import numpy as np
import pandas as pd
import re
import tensorflow as tf
from keras import layers

def standardize(val):
    return re.sub(r',|\+|\(|\)|\d|\.|/|-', ' ', val).lower().split()

def standardize_y(rawData):
    d = {}
    idx = 0
    for i in range(len(rawData)):
        r = rawData[i]
        y = rawData[i][1]       
        if y not in d:
            d[y] = idx
            
        rawData[i][1] = d[y]
        idx += 1

    return {v: k for k, v in d.items()}



def split_data(rawData):
    num = rawData.shape[0]
    train_num = int(0.7 * num)
    test_num = int(0.2 * num)
    return [
        (rawData[:train_num, 0], rawData[:train_num, 1]),
        (rawData[train_num:train_num + test_num, 0], rawData[train_num:train_num + test_num, 1]),
        (rawData[train_num + test_num:, 0], rawData[train_num + test_num:, 1]),
    ]

def create_vocabulary(rawData):
    voc = set()

    for s in rawData[:, 0]:
        for w in standardize(s):
            if w != '':
                voc.add(w)
    
    return list(voc)