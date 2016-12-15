import pandas as pd #数据分析
import numpy as np #科学计算
from pandas import Series,DataFrame
import matplotlib.pyplot as plt
from sklearn.learning_curve import learning_curve
from sklearn.ensemble import RandomForestRegressor
import sklearn.preprocessing as preprocessing
from sklearn import linear_model
from sklearn.ensemble import BaggingRegressor
from sklearn.ensemble import RandomForestClassifier


train = pd.read_csv("data/train.csv")
test = pd.read_csv("data/test.csv")

def set_missing_ages(df):
    
    # 把已有的数值型特征取出来丢进Random Forest Regressor中
    age_df = df[['Age','Fare', 'Parch', 'SibSp', 'Pclass']]

    # 乘客分成已知年龄和未知年龄两部分
    known_age = age_df[age_df.Age.notnull()].as_matrix()
    unknown_age = age_df[age_df.Age.isnull()].as_matrix()

    # y即目标年龄
    y = known_age[:, 0]

    # X即特征属性值
    X = known_age[:, 1:]

    # fit到RandomForestRegressor之中
    rfr = RandomForestRegressor(random_state=0, n_estimators=2000, n_jobs=-1)
    rfr.fit(X, y)
    
    # 用得到的模型进行未知年龄结果预测
    predictedAges = rfr.predict(unknown_age[:, 1::])
    
    # 用得到的预测结果填补原缺失数据
    df.loc[ (df.Age.isnull()), 'Age' ] = predictedAges 
    
    return df


def set_Cabin_type(df):
    df.loc[ (df.Fare.isnull()), 'Fare' ] = 0
    df.loc[df.Cabin.notnull(), 'Cabin' ] = "Yes"
    df.loc[df.Cabin.isnull(), 'Cabin' ] = "No"

    df.loc[df.Embarked.isnull(), 'Embarked' ] = "S"
    return df


def dummy(data_train):
    dummies_Cabin = pd.get_dummies(data_train['Cabin'], prefix= 'Cabin')

    dummies_Embarked = pd.get_dummies(data_train['Embarked'], prefix= 'Embarked')

    dummies_Sex = pd.get_dummies(data_train['Sex'], prefix= 'Sex')

    dummies_Pclass = pd.get_dummies(data_train['Pclass'], prefix= 'Pclass')

    df = pd.concat([data_train, dummies_Cabin, dummies_Embarked, dummies_Sex, dummies_Pclass], axis=1)
    df.drop(['Pclass', 'Name', 'Sex', 'Ticket', 'Cabin', 'Embarked'], axis=1, inplace=True)
    return df


def scale(df):
    scaler = preprocessing.StandardScaler()
    age_scale_param = scaler.fit(df['Age'])
    df['Age_scaled'] = scaler.fit_transform(df['Age'], age_scale_param)
    fare_scale_param = scaler.fit(df['Fare'])
    df['Fare_scaled'] = scaler.fit_transform(df['Fare'], fare_scale_param)
    return df


def clean(df):
    df = set_Cabin_type(df)
    df = set_missing_ages(df)
    df = dummy(df)
    df = scale(df)
    df = df.filter(regex='Survived|Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*')
    return df

def lg(df, test_df):
    train_np = train_df.as_matrix()
    # y即Survival结果
    y = train_np[:, 0]
    # X即特征属性值
    X = train_np[:, 1:]
    # fit到RandomForestRegressor之中
    clf = linear_model.LogisticRegression(C=1.0, penalty='l1', tol=1e-6)
    clf.fit(X, y)

    predictions = clf.predict(test_df)
    result = pd.DataFrame({'PassengerId':test['PassengerId'].as_matrix(), 'Survived':predictions.astype(np.int32)})
    return result, clf, X, y


def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None, n_jobs=1, 
                        train_sizes=np.linspace(.05, 1., 20), verbose=0, plot=True):
    """
    画出data在某模型上的learning curve.
    参数解释
    ----------
    estimator : 你用的分类器。
    title : 表格的标题。
    X : 输入的feature，numpy类型
    y : 输入的target vector
    ylim : tuple格式的(ymin, ymax), 设定图像中纵坐标的最低点和最高点
    cv : 做cross-validation的时候，数据分成的份数，其中一份作为cv集，其余n-1份作为training(默认为3份)
    n_jobs : 并行的的任务数(默认1)
    """
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes, verbose=verbose)
    
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    
    if plot:
        plt.figure()
        plt.title(title)
        if ylim is not None:
            plt.ylim(*ylim)
        plt.xlabel(u"训练样本数")
        plt.ylabel(u"得分")
        plt.gca().invert_yaxis()
        plt.grid()
    
        plt.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, 
                         alpha=0.1, color="b")
        plt.fill_between(train_sizes, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, 
                         alpha=0.1, color="r")
        plt.plot(train_sizes, train_scores_mean, 'o-', color="b", label=u"训练集上得分")
        plt.plot(train_sizes, test_scores_mean, 'o-', color="r", label=u"交叉验证集上得分")
    
        plt.legend(loc="best")
        
        plt.draw()
        plt.gca().invert_yaxis()
        plt.show()
    
    midpoint = ((train_scores_mean[-1] + train_scores_std[-1]) + (test_scores_mean[-1] - test_scores_std[-1])) / 2
    diff = (train_scores_mean[-1] + train_scores_std[-1]) - (test_scores_mean[-1] - test_scores_std[-1])
    return midpoint, diff


def bagging(X, y):
    clf = linear_model.LogisticRegression(C=1.0, penalty='l1', tol=1e-6)
    bagging_clf = BaggingRegressor(clf, n_estimators=10, max_samples=0.8, max_features=1.0, bootstrap=True, bootstrap_features=False, n_jobs=-1)
    bagging_clf.fit(X, y)

    predictions = bagging_clf.predict(test_df)
    result = pd.DataFrame({'PassengerId':test['PassengerId'].as_matrix(), 'Survived':predictions.astype(np.int32)})
    result.to_csv("data/ensemble.csv", index=False)

train_df = clean(train)
test_df = clean(test)
result, clf, X, y = lg(train_df, test_df)
# plot_learning_curve(clf, u"学习曲线", X, y)
pd.DataFrame({"columns":list(train_df.columns)[1:], "coef":list(clf.coef_.T)})
bagging(X, y)
