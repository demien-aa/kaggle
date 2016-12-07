# https://www.kaggle.com/omarelgabry/titanic/a-journey-through-titanic
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

from etl import ETL
from plot import CPlot, DPlot, show


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

    def __init__(self):
        super(Family, self).__init__()
        self.tdf['Family'] = self.tdf['Parch'] + self.tdf['SibSp']


# Embarked().profile()
# Sex().profile()
# PClass().profile()
# Fare().profile()
# Age().profile()
Family().profile()
