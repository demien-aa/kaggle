import json
import numpy as np
import pandas as pd
import re
import string
import warnings

from common.util import submit_stamp, red, green, white, blue, single_line
from operator import itemgetter
from pandas import DataFrame
from patsy import dmatrices
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline, make_pipeline


warnings.filterwarnings('ignore')


train_file = "data/train.csv"
test_file = "data/test.csv"
seed = 0


def main():
    '''
    The main workflow, entry point
    '''
    train_df = clean(pd.read_csv(train_file))
    estimator = grid_search(train_df)
    predict(estimator, True)


def clean(df):
    print(white(single_line('Clean Data Frame')))
    df = embarked(df)
    df = gender(df)
    df = family(df)
    df = title(df)
    df = fare(df)
    df = age_by_rf(df)
    df = age_class(df)
    df = dummy(df, 'Embarked')
    df = dummy(df, 'Title')
    df = dummy(df, 'AgeClass')
    df = df.drop(['PassengerId', 'Name', 'Age', 'Cabin', 'Sex', 'Ticket'], axis=1)
    print(blue('Finish Clean'))
    return df


def dummy(df, name):
    df = pd.concat([df, pd.get_dummies(df[name], prefix=name)], axis=1)
    return df.drop(name, axis=1)


def age_by_rf(df):
    new_df = pd.concat([df, pd.get_dummies(df.Title, prefix='Title')], axis=1)
    # Get all data to random forest regression
    age_df = new_df[['Age','Fare', 'Parch', 'Family', 'Gender', 'Title_Master', 'Title_Miss', 'Title_Mr', 'Title_Mrs']]

    train = age_df[age_df.Age.notnull()].as_matrix()
    predict = age_df[age_df.Age.isnull()].as_matrix()

    # Train
    rfr = RandomForestRegressor(random_state=0, n_estimators=2000, n_jobs=-1)
    rfr.fit(train[:, 1:], train[:, 0])

    # Predict
    df.loc[df.Age.isnull(), 'Age'] = rfr.predict(predict[:, 1:])
    return df


def embarked(df):
    df.Embarked.fillna('S')
    return df


def gender(df):
    df['Gender'] = df['Sex'].map({'female': 0, 'male': 1}).astype(int)
    return df


def family(df):
    df['Family'] = df['SibSp'] + df['Parch']
    return df


def fare(df):
    # change 0 to nan for easy handle
    df.Fare = df.Fare.map(lambda x: np.nan if x == 0 else x)
    df.loc[df.Fare.isnull() & (df.Pclass == 1), 'Fare'] = df[df.Pclass == 1].Fare.median()
    df.loc[df.Fare.isnull() & (df.Pclass == 2), 'Fare'] = df[df.Pclass == 2].Fare.median()
    df.loc[df.Fare.isnull() & (df.Pclass == 3), 'Fare']= df[df.Pclass == 3].Fare.median()
    return df


def title(df):
    # 'Mr', 'Mrs', 'Miss', 'Master', 'Don', 'Rev', 'Dr', 'Mme', 'Ms', 'Major', 'Lady', 'Sir', 'Mlle', 'Col', 'Capt', 'the Countess', 'Jonkheer'
    pattern = '.*,( .*?)\. .*'
    title_list = df.Name.str.extract(pattern).map(lambda x: x.strip()).unique()

    def get_title(row):
        name = row['Name']
        title = re.findall(pattern, name)
        if title:
            title = title[0].strip()
            if title in ['Mr','Don', 'Major', 'Capt', 'Jonkheer', 'Rev', 'Col', 'Sir']: # Man
                return 'Mr'
            elif title in ['Master']: # Young man or boy
                return 'Master'
            elif title in ['the Countess', 'Mme', 'Mrs', 'Lady']: # Married women
                return 'Mrs'
            elif title in ['Mlle', 'Ms','Miss']: # Young women
                return 'Miss'
            else: # Others check gender
                print(red('Unknown title %s' % title))
                if row['Sex'] == 'Male':
                    return 'Mr'
                else:
                    return 'Mrs'
        else:
            raise('Could not extract title %s' % name)
            return np.nan
        
    df['Title'] = df.apply(get_title, axis=1)
    return df


def age_class(df):
    df['AgeClass'] = df['Age']
    df.loc[(df.Age <= 10), 'AgeClass'] = 'child'
    df.loc[(df.Age > 10) & (df.Age <= 30) ,'AgeClass'] = 'adult'
    df.loc[(df.Age > 30) & (df.Age <= 60) ,'AgeClass'] = 'senior'
    df.loc[(df.Age > 60), 'AgeClass'] = 'aged'
    return df


def rfc():
    return RandomForestClassifier(n_estimators=500, criterion='entropy', max_depth=5, min_samples_split=2, min_samples_leaf=1, max_features='auto', bootstrap=False, oob_score=False, n_jobs=1, random_state=seed, verbose=0)


def lgr():
    return LogisticRegression(C=1.0, penalty='l1', tol=1e-6)


def grid_search(df):
    print(white(single_line('Train Grid Search')))
    # grid seach and find the best estimator
    x = df.drop('Survived', axis=1)
    y = df.Survived
    pipeline = make_pipeline(rfc(), lgr())
    grid_search = grid_search = GridSearchCV(pipeline, param_grid={}, verbose=3, cv=30, scoring='accuracy').fit(x, y)
    print(green('Best score: %0.3f' % grid_search.best_score_))
    grid_search_report(grid_search.grid_scores_)
    print('Best estimator: ', grid_search.best_estimator_)
    print(blue('Finish Grid Search'))
    return grid_search.best_estimator_


def predict(estimator, csv=False):
    print(white(single_line('Predict Test Data Set')))
    test_raw = pd.read_csv(test_file)
    test_df = clean(test_raw)
    survived = estimator.predict(test_df).astype(np.int32)
    result = pd.DataFrame({'PassengerId': test_raw['PassengerId'].as_matrix(), 'Survived': survived})
    if csv:
        print(blue('Save CSV'))
        result.to_csv('result/solution-2_%s.csv' % submit_stamp(), index=False)
    print(blue('Finish Predict'))


def grid_search_report(grid_scores, n_top=3):
    top_scores = sorted(grid_scores, key=itemgetter(1), reverse=True)[:n_top]
    for i, score in enumerate(top_scores):
        print('Mean validation score: {0:.3f} (std: {1:.3f})'.format(
              score.mean_validation_score,
              np.std(score.cv_validation_scores)))


if __name__ == '__main__':
    main()
