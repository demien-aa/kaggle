# https://www.kaggle.com/omarelgabry/titanic/a-journey-through-titanic

import pandas as pd

from pandas import Series,DataFrame

# numpy, matplotlib, seaborn
import numpy as np
import seaborn as sns
sns.set_style('whitegrid')
import matplotlib
# Force matplotlib to not use any Xwindows backend.
import matplotlib.pyplot as plt
# %matplotlib inline

# machine learning
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB


from plot import CPlot, DPlot, show
from etl import ETL


class Embarked(DPlot):
    NAME = 'Embarked'


class Sex(DPlot):
    NAME = 'Sex'


class PClass(DPlot):
    NAME = 'Pclass'


class Fare(CPlot):
    NAME = 'Fare'
    HIST = 30


class Age(CPlot, ETL):
    NAME = 'Age'
    HIST = 70

    def __init__(self):
        super(Age, self).__init__()
        self.fill_na_by_mean()


class Family(CPlot):
    NAME = 'Family'
    HIST = 10


Embarked().profile()
Sex().profile()
PClass().profile()
Fare().profile()
Age().profile()
Family().profile()
