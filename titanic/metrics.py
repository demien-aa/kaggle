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

    @show
    def profile(self):
        self.count()
        self.mean_bar()
        self.point()


class Fare(CPlot):
    NAME = 'Fare'

    def profile(self):
        self.hist()
        self.mean_bar()


class Age(CPlot, ETL):
    NAME = 'Age'

    def __init__(self):
        super(Age, self).__init__()
        self.fill_na_by_mean()

    def profile(self):
        self.hist(70)
        self.mean_bar_by_name()


# Embarked().profile()
# Fare().profile()
Age().profile()
