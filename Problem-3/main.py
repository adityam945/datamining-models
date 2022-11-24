import csv
import numpy as np

from sklearn.model_selection import train_test_split
# from Mo import *
from Models import *
from DataProcessing import *
import random
import warnings
warnings.filterwarnings('always')
# 
def switchModels(x_not_fraud, x_fraud, y_not_fraud, y_fraud):
    # list_preds = 
        # 1
        score1, cls_rpt1, score_log1, cls_rpt_log1 = modelOne(x_not_fraud, x_fraud, y_not_fraud, y_fraud)
        score2, cls_rpt2 = modelTwo(x_not_fraud, x_fraud, y_not_fraud, y_fraud)
        score3, cls_rpt3 = modelThree(x_not_fraud, x_fraud, y_not_fraud, y_fraud)
        score4, cls_rpt4 = modeFour(x_not_fraud, x_fraud, y_not_fraud, y_fraud)
# 
        print("model - 1 scored an accuracy\nLinearRegression ",score1,'\n', cls_rpt1, 'on the raw dataset')
        print("model - 1 scored an accuracy\nLogisticRegression ",score_log1, '\n',cls_rpt_log1, 'on the raw dataset')
        # 2
        # x_fraud, y_fraud mutliple by 10 times
        print("model - 2 scored an accuracy",score2,'\n', cls_rpt2 , 'on data increased tile')
        # 3
        # x_fraud, y_fraud mutliple by 10 times and multiply by some weights random
        #  [0.1, 0.05, 0.2, 0.015, 0.001]
        print("model - 3 scored an accuracy",score3,'\n', cls_rpt3 , 'weight multiply')
        # 3
        # x_fraud, y_fraud mutliple by 10 times and multiply by some weights random
        #  [0.1, 0.05, 0.2, 0.015, 0.001]
        print("model - 4 scored an accuracy",score4,'\n', cls_rpt4 , 'SMOTE Upsampling')




if __name__ == '__main__':
    x_not_fraud, x_fraud, y_not_fraud, y_fraud = initalData('./data/creditcard.csv')
    switchModels(x_not_fraud, x_fraud, y_not_fraud, y_fraud)
