import json
import numpy as np
import pandas as pd
import re
import string

from common.util import submit_stamp
from operator import itemgetter
from pandas import DataFrame
from patsy import dmatrices
from sklearn import preprocessing
from sklearn.cross_validation import cross_val_score, train_test_split, StratifiedShuffleSplit, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.externals import joblib
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline


train_file = "data/train.csv"
test_file = "data/test.csv"
seed = 0


def main():
    # The main workflow, entry point

    # Clean and feature engineering
    train_df = clean(pd.read_csv(train_file))



def clean(df):
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
                print('Unknown title %s' % title)
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


if __name__ == '__main__':
    main()

##Read configuration parameters



# print(train_file,seed)

# # 输出得分
# def report(grid_scores, n_top=3):
#     top_scores = sorted(grid_scores, key=itemgetter(1), reverse=True)[:n_top]
#     for i, score in enumerate(top_scores):
#         print('Model with rank: {0}'.format(i + 1))
#         print('Mean validation score: {0:.3f} (std: {1:.3f})'.format(
#               score.mean_validation_score,
#               np.std(score.cv_validation_scores)))
#         print('Parameters: {0}'.format(score.parameters))

# #清理和处理数据
# def substrings_in_string(big_string, substrings):
#     for substring in substrings:
#         if big_string.find(substring) != -1:
#             return substring
#     print(big_string)
#     return np.nan




def clean_and_munge_data(df):


    # df['AgeFill']=df['Age']
    # mean_ages = np.zeros(4)
    # mean_ages[0]=np.average(df[df['Title'] == 'Miss']['Age'].dropna())
    # mean_ages[1]=np.average(df[df['Title'] == 'Mrs']['Age'].dropna())
    # mean_ages[2]=np.average(df[df['Title'] == 'Mr']['Age'].dropna())
    # mean_ages[3]=np.average(df[df['Title'] == 'Master']['Age'].dropna())
    # df.loc[ (df.Age.isnull()) & (df.Title == 'Miss') ,'AgeFill'] = mean_ages[0]
    # df.loc[ (df.Age.isnull()) & (df.Title == 'Mrs') ,'AgeFill'] = mean_ages[1]
    # df.loc[ (df.Age.isnull()) & (df.Title == 'Mr') ,'AgeFill'] = mean_ages[2]
    # df.loc[ (df.Age.isnull()) & (df.Title == 'Master') ,'AgeFill'] = mean_ages[3]



    df['Fare_Per_Person']=df['Fare']/(df['Family_Size']+1)

    #Age times class

    df['AgeClass']=df['AgeFill']*df['Pclass']
    df['ClassFare']=df['Pclass']*df['Fare_Per_Person']


    df['HighLow']=df['Pclass']
    df.loc[ (df.Fare_Per_Person<8) ,'HighLow'] = 'Low'
    df.loc[ (df.Fare_Per_Person>=8) ,'HighLow'] = 'High'



    le.fit(df['Sex'] )
    x_sex=le.transform(df['Sex'])
    df['Sex']=x_sex.astype(np.float)

    le.fit( df['Ticket'])
    x_Ticket=le.transform( df['Ticket'])
    df['Ticket']=x_Ticket.astype(np.float)

    le.fit(df['Title'])
    x_title=le.transform(df['Title'])
    df['Title'] =x_title.astype(np.float)

    le.fit(df['HighLow'])
    x_hl=le.transform(df['HighLow'])
    df['HighLow']=x_hl.astype(np.float)


    le.fit(df['AgeCat'])
    x_age=le.transform(df['AgeCat'])
    df['AgeCat'] =x_age.astype(np.float)

    le.fit(df['Embarked'])
    x_emb=le.transform(df['Embarked'])
    df['Embarked']=x_emb.astype(np.float)

    df = df.drop(['PassengerId','Name','Age','Cabin'], axis=1) #remove Name,Age and PassengerId


    return df

# #读取数据
# traindf=pd.read_csv(train_file)
# ##清洗数据
# df=clean_and_munge_data(traindf)
# ########################################formula################################
# # import ipdb; ipdb.set_trace() 
# formula_ml='Survived~Pclass+C(Title)+Sex+C(AgeCat)+Fare_Per_Person+Fare+Family_Size' 
# # import ipdb; ipdb.set_trace()
# y_train, x_train = dmatrices(formula_ml, data=df, return_type='dataframe')
# y_train = np.asarray(y_train).ravel()
# print(y_train.shape,x_train.shape)

# ##选择训练和测试集
# X_train, X_test, Y_train, Y_test = train_test_split(x_train, y_train, test_size=0.2,random_state=seed)
# #初始化分类器
# clf=RandomForestClassifier(n_estimators=500, criterion='entropy', max_depth=5, min_samples_split=2,
#   min_samples_leaf=1, max_features='auto',    bootstrap=False, oob_score=False, n_jobs=1, random_state=seed,
#   verbose=0)

# ###grid search找到最好的参数
# param_grid = dict( )
# ##创建分类pipeline
# pipeline=Pipeline([ ('clf',clf) ])
# # grid_search = GridSearchCV(pipeline, param_grid=param_grid, verbose=3,scoring='accuracy',cv=StratifiedShuffleSplit(Y_train, n_iter=10, test_size=0.2, train_size=None, indices=None, random_state=seed, n_iterations=None)).fit(X_train, Y_train)
# grid_search = GridSearchCV(pipeline, param_grid=param_grid, verbose=3, cv=3, scoring='accuracy').fit(X_train, Y_train)
# # 对结果打分
# print('Best score: %0.3f' % grid_search.best_score_)
# print(grid_search.best_estimator_)
# report(grid_search.grid_scores_)
 
# print('-----grid search end------------')
# print ('on all train set')
# scores = cross_val_score(grid_search.best_estimator_, x_train, y_train,cv=3,scoring='accuracy')
# print(scores.mean(),scores)
# print ('on test set')
# scores = cross_val_score(grid_search.best_estimator_, X_test, Y_test,cv=3,scoring='accuracy')
# print(scores.mean(),scores)

# # 对结果打分

# print(classification_report(Y_train, grid_search.best_estimator_.predict(X_train) ))
# print('test data')
# print(classification_report(Y_test, grid_search.best_estimator_.predict(X_test) ))


# #读取数据
# testdf=pd.read_csv(test_file)
# ##清洗数据
# testnp=clean_and_munge_data(testdf)
# formula_ml='HighLow~Pclass+C(Title)+Sex+C(AgeCat)+Fare_Per_Person+Fare+Family_Size'
# y, x = dmatrices(formula_ml, data=testnp, return_type='dataframe')
# result = pd.DataFrame({'PassengerId':testdf['PassengerId'].as_matrix(), 'Survived':grid_search.best_estimator_.predict(x).astype(np.int32)})
# result.to_csv('result/solution_2_%s.csv' % submit_stamp(), index=False)
