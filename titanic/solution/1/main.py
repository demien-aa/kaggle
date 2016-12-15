import pandas as pd
import numpy as np
import datetime
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

train = pd.read_csv("data/train.csv")
test = pd.read_csv("data/test.csv")

def foo(train):
    train['Family'] =  train["Parch"] + train["SibSp"]
    train = train.drop(['Parch','SibSp'], axis=1)

    train["Embarked"].fillna("S")
    train = train.join(pd.get_dummies(train['Embarked']))
    train = train.drop('Embarked', axis=1)

    train = train.drop('S', axis=1)
    train["Fare"].fillna(train["Fare"].median(), inplace=True)

    age_mean = train["Age"].mean()
    age_std = train["Age"].std()
    age_null_cnt = train["Age"].isnull().sum()
    age_rand = np.random.randint(age_mean - age_std, age_mean + age_std, size = age_null_cnt)
    train["Age"][np.isnan(train["Age"])] = age_rand

    train = train.join(pd.get_dummies(train['Sex']))
    train = train.drop('Sex', axis=1)
    train = train.drop(['PassengerId','Name','Ticket', 'Cabin'], axis=1)

    def check_child(p):
        return 1 if p.Age < 6 else 0
        
    train['Child'] = train.apply(check_child, axis=1)
    return train

train = foo(train)
train_x = train.drop('Survived', axis=1)
train_y = train['Survived']
_test = foo(test)

lg = LogisticRegression()
lg.fit(train_x, train_y)
print('lg score', lg.score(train_x, train_y))

coeff_df = pd.DataFrame(train_x.columns.delete(0))
coeff_df.columns = ['Features']
coeff_df["Coefficient Estimate"] = pd.Series(lg.coef_[0])

random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(train_x, train_y)
print('rf score', random_forest.score(train_x, train_y))

_text_y = random_forest.predict(_test)
rf_test = test.join(pd.DataFrame(_text_y, columns=['Survived']))
rf_test[['PassengerId', 'Survived']].to_csv('result/solution_1_ramdom-forest_%s.csv' % datetime.datetime.now().strftime('%Y%m%d_%H%M%S'), index=False)

_text_y = lg.predict(_test)
lg_test = test.join(pd.DataFrame(_text_y, columns=['Survived']))
lg_test[['PassengerId', 'Survived']].to_csv('result/solution_1_lg_%s.csv' % datetime.datetime.now().strftime('%Y%m%d_%H%M%S'), index=False)
