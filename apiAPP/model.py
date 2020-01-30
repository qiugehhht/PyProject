import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

# read data and data split
df = pd.read_csv('df.csv')
dfdrop = df.drop(['c_temp','c_eda','c_emg'],axis=1)
y = dfdrop['y']
X = dfdrop.iloc[:,1:]
X_train, X_test, Y_train, Y_test = train_test_split(X, y )

# function of display data
def read_data():
    try:
        data = df.head(50)
        return data
    except RuntimeError:
        return "Wrong Path"
# Decision tree grid search
def decision_tree_algo():
    algorithm = DecisionTreeClassifier()
    hyperparameters = {
        "max_depth"    : [5, 10,20],
        "max_features" : [5, 8, 10]
    }

    grid_optimization = GridSearchCV(algorithm, 
                                         hyperparameters, 
                                         cv=5, 
                                         iid=False, )

    scores = grid_optimization.fit(X_train, Y_train)
    return scores

#random forest grid search
def random_tree_algo():
    algorithm1 = RandomForestClassifier()
    hyperparameters1 = {
        "n_estimators" : [10, 20, 40],
        "max_depth"    : [5, 10,20],
        "max_features" : [5, 8, 10]
    }

    grid_optimization = GridSearchCV(algorithm1, 
                                         hyperparameters1, 
                                         cv=5, 
                                         iid=False, )

    scores1 = grid_optimization.fit(X_train, Y_train)
    return scores1

# logistic regression
def log_regression_algo():
    lrmodel = LogisticRegression(random_state=0).fit(X_train, Y_train)
    score3 = lrmodel.score(X_test, Y_test)
    return score3

# display heatmap
def heatplt():
    plt = sns.heatmap(df.corr(), square=True, annot=True)
    return plt

# inside function for accuracy assessment
def get_score(algorithme, X_train, X_test, Y_train, Y_test):
    modele     = algorithme.fit(X_train, Y_train)
    score      = modele.score(X_test, Y_test)
    return score

# build user defined random forest model, and provide this model for prediction 
def pre_random_m(ntrees,mtry,depth):
    algorithme = RandomForestClassifier(n_estimators=ntrees, max_features=mtry, max_depth=depth)
    score = get_score(algorithme, X_train, X_test, Y_train, Y_test)
    modelran = algorithme.fit(X_train, Y_train)
    final = (score,modelran)
    return final