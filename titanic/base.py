import pandas as pd


class Base(object):
    TARGET = 'Survived'

    def __init__(self):
        # load train data frame and predict data frame
        self.tdf = pd.read_csv("data/train.csv")
        self.pdf = pd.read_csv("data/test.csv")
