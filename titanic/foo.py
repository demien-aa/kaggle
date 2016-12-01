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


class Factor(object):
    TARGET = 'Survived'

    def __init__(self):
        # load train data frame and predict data frame
        self.tdf = pd.read_csv("data/train.csv")
        self.pdf = pd.read_csv("data/test.csv")


class Embarked(Factor):
    NAME = 'Embarked'

    def countplot(self):
        _, (axis1,axis2,axis3) = plt.subplots(1,3,figsize=(15,5))
        sns.countplot(x=self.NAME, data=self.tdf, ax=axis1)
        sns.countplot(x=self.TARGET, hue=self.NAME, data=self.tdf, ax=axis2)
        embark_perc = self.tdf[[self.NAME, self.TARGET]].groupby([self.NAME], as_index=False).mean()
        sns.barplot(x=self.NAME, y=self.TARGET, data=embark_perc, ax=axis3)

    def factorplot(self):
        sns.factorplot(self.NAME, self.TARGET, data=self.tdf, size=4, aspect=3)

    def profile(self):
        self.countplot()
        self.factorplot()

    def get_t_dummies(self):
        return pd.get_dummies(self.tdf[self.NAME]).drop(['S'], axis=1)

    def get_p_dummies(self):
        return pd.get_dummies(self.pdf[self.NAME]).drop(['S'], axis=1)


class Fare(Factor):
    NAME = 'Fare'

    def hist(self):
        self.tdf[self.NAME].plot(kind='hist', bins=100)

    def plot_mean(self):
        sns.barplot(x=self.TARGET, y=self.NAME, data=self.tdf[[self.NAME, self.TARGET]].groupby([self.TARGET], as_index=False).mean())
        # plt.plot()

    def profile(self):
        self.hist()
        self.plot_mean()


# Embarked().profile()
Fare().profile()
plt.show()


# # get titanic & test csv files as a DataFrame
# titanic_df = pd.read_csv("data/train.csv")
# test_df    = pd.read_csv("data/test.csv")

# # preview the data
# titanic_df.info()
# test_df.info()

# # drop unnecessary columns, these columns won't be useful in analysis and prediction
# titanic_df = titanic_df.drop(['PassengerId','Name','Ticket'], axis=1)
# test_df    = test_df.drop(['Name','Ticket'], axis=1)


# # Embarked

# # only in titanic_df, fill the two missing values with the most occurred value, which is "S".
# titanic_df["Embarked"] = titanic_df["Embarked"].fillna("S")

# # plot
# sns.factorplot('Embarked','Survived', data=titanic_df,size=4,aspect=3)

# fig, (axis1,axis2,axis3) = plt.subplots(1,3,figsize=(15,5))
# # sns.factorplot('Embarked',data=titanic_df,kind='count',order=['S','C','Q'],ax=axis1)
# # sns.factorplot('Survived',hue="Embarked",data=titanic_df,kind='count',order=[1,0],ax=axis2)
# sns.countplot(x='Embarked', data=titanic_df, ax=axis1)
# sns.countplot(x='Survived', hue="Embarked", data=titanic_df, order=[1,0], ax=axis2)


# # group by embarked, and get the mean for survived passengers for each value in Embarked
# embark_perc = titanic_df[["Embarked", "Survived"]].groupby(['Embarked'],as_index=False).mean()
# sns.barplot(x='Embarked', y='Survived', data=embark_perc,order=['S','C','Q'],ax=axis3)


# # Either to consider Embarked column in predictions,
# # and remove "S" dummy variable, 
# # and leave "C" & "Q", since they seem to have a good rate for Survival.

# # OR, don't create dummy variables for Embarked column, just drop it, 
# # because logically, Embarked doesn't seem to be useful in prediction.

# # embark_dummies_titanic  = pd.get_dummies(titanic_df['Embarked'])
# # embark_dummies_titanic.drop(['S'], axis=1, inplace=True)

# # embark_dummies_test  = pd.get_dummies(test_df['Embarked'])
# # embark_dummies_test.drop(['S'], axis=1, inplace=True)

# # titanic_df = titanic_df.join(embark_dummies_titanic)
# # test_df    = test_df.join(embark_dummies_test)

# # titanic_df.drop(['Embarked'], axis=1,inplace=True)
# # test_df.drop(['Embarked'], axis=1,inplace=True)

# plt.show()
