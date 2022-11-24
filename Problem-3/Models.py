from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics 
from DataProcessing import *
from sklearn.metrics import confusion_matrix, classification_report
from imblearn.over_sampling import SMOTE

import random


def Cal_LR(X_train, X_test, y_train, y_test):
    regr = LinearRegression()
  
    regr.fit(X_train, y_train)
    predictions = regr.predict(X_test)
    predictions = np.absolute(np.round(predictions, 2))
    predictions = (np.ceil(predictions)).astype(int)
    score = regr.score(X_test, y_test)
    cls_rpt = classification_report(y_test, predictions, labels=np.unique(predictions))

    return score, cls_rpt

def Cal_Loges_R(X_train, X_test, y_train, y_test):
    regr = LogisticRegression()
  
    regr.fit(X_train, y_train)
    predictions = regr.predict(X_test)
    predictions = np.absolute(np.round(predictions, 2))
    predictions = (np.ceil(predictions)).astype(int)
    score = regr.score(X_test, y_test)
    cls_rpt = classification_report(y_test, predictions, labels=np.unique(predictions))

    return score, cls_rpt

#  random forest
def smote_upsample_data(X_train, X_test, y_train, y_test):
    sm = SMOTE(random_state = 2)
    X_train_res, y_train_res = sm.fit_sample(X_train, y_train)
    lr_upsample = LogisticRegression()
    lr_upsample.fit(X_train_res, y_train_res.ravel())
    predictions = lr_upsample.predict(X_test)
    score = lr_upsample.score(X_test, y_test)
    # print classification report
    cls_rpt = classification_report(y_test, predictions)
    return score, cls_rpt


# custom
def modelOne(x_not_fraud, x_fraud, y_not_fraud, y_fraud):
    x_data = concat_arrays(x_not_fraud, x_fraud, 0)
    y_labels = concat_arrays(y_not_fraud, y_fraud, 0)
    X_train, X_test, y_train, y_test = train_test_split(x_data, y_labels, test_size = 0.3)
    score, cls_rpt = Cal_LR(X_train, X_test, y_train, y_test)
    score_log, cls_rpt_log = Cal_Loges_R(X_train, X_test, y_train, y_test)

    return score, cls_rpt, score_log, cls_rpt_log


def modelTwo(x_not_fraud, x_fraud, y_not_fraud, y_fraud):
    x_fraud = npTileArray(x_fraud, 10)
    y_fraud = npTileArrayOneDimension(y_fraud, 10) 

    x_data = concat_arrays(x_not_fraud, x_fraud, 0)
    y_labels = concat_arrays(y_not_fraud, y_fraud, 0)

    X_train, X_test, y_train, y_test = train_test_split(x_data, y_labels, test_size = 0.3)

    score, cls_rpt = Cal_Loges_R(X_train, X_test, y_train, y_test)
    # print(score)
    return score, cls_rpt

def modelThree(x_not_fraud, x_fraud, y_not_fraud, y_fraud):
    # create copies of original
    x_fraudcopy = x_fraud
    y_fraudcopy = y_fraud
    # random w list
    random_w_list = [1.1, 1.2, 1.3]
    # x_fraud, y_fraud mutliple by 10 times
    x_fraud = npTileArray(x_fraud, 10)
    y_fraud = npTileArrayOneDimension(y_fraud, 10) 
    # multiple weights for each dimensions
    for i in range(np.array(x_fraud).shape[0]):
        random_w = random.choice(random_w_list)
        for j in range(np.array(x_fraud).shape[1]):
            # random and multiply
            x_fraud[i][j] = x_fraud[i][j] * convertListToNpArray(random_w)
    # concatinate
    x_fraud = concat_arrays(x_fraudcopy ,x_fraud, 0)
    y_fraud = concat_arrays(y_fraudcopy ,y_fraud, 0) 
    x_data = concat_arrays(x_not_fraud, x_fraud, 0)
    y_labels = concat_arrays(y_not_fraud, y_fraud, 0)
    # 
    X_train, X_test, y_train, y_test = train_test_split(x_data, y_labels, test_size = 0.3)

    score, cls_rpt = Cal_Loges_R(X_train, X_test, y_train, y_test)
    # print(score)
    return score, cls_rpt

def modeFour(x_not_fraud, x_fraud, y_not_fraud, y_fraud):
    x_fraud = npTileArray(x_fraud, 10)
    y_fraud = npTileArrayOneDimension(y_fraud, 10) 

    x_data = concat_arrays(x_not_fraud, x_fraud, 0)
    y_labels = concat_arrays(y_not_fraud, y_fraud, 0)

    X_train, X_test, y_train, y_test = train_test_split(x_data, y_labels, test_size = 0.3)

    score, cls_rpt = smote_upsample_data(X_train, X_test, y_train, y_test)
    # print(score)
    return score, cls_rpt