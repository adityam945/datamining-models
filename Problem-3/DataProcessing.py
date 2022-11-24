import csv
import numpy as np
from sklearn.model_selection import train_test_split

# inital data preprocessing
def initalData(path):
    # 
    print("Extracting data .....%%%%%")
    creditcard_data = readFile(path)
    print("Extracting data completed \nPreprocessing Data")
    # delete first row
    creditcard_data = deleteFirstRow(creditcard_data)
    # 
    creditcard_data_non_fraud, creditcard_data_fraud = spit_fraud_data(creditcard_data)
    # convert to float
    creditcard_data_non_fraud = convertStrToFloat(creditcard_data_non_fraud)
    creditcard_data_fraud = convertStrToFloat(creditcard_data_fraud)
    # seperate data and labels
    # get last row the y labels
    # fraud
    x_fraud, y_fraud = seperate_labels_data(creditcard_data_fraud)
    # not fraud
    x_not_fraud, y_not_fraud = seperate_labels_data(creditcard_data_non_fraud)
    # 
    print('Data preprocessing completed!')
    # models cases from here
    return x_not_fraud, x_fraud, y_not_fraud, y_fraud
# 
def readFile(path):
    data = []
    with open(path, mode ='r')as file:
        csvFile = csv.reader(file)
        for lines in csvFile:
                data.append(lines)
    return data

# 
def seperate_labels_data(lst):
    listItems_y = [item[-1] for item in lst]
    # convert to int from float
    y = [int(x) for x in listItems_y]
    x = [item[:-1] for item in lst]
    return x, y


# 
def deleteFirstRow(list):
    return np.delete(list, 0, axis=0)

# 
def convertStrToFloat(lst):
    return [list( map(float,i) ) for i in lst]

# 
def convertListToNpArray(lst):
    return np.array(lst)

# 
def spit_fraud_data(lst):
    fraud_data = []
    non_fraud_data = []
    for i in range(convertListToNpArray(lst).shape[0]):
        if lst[i][30] == '0':
            non_fraud_data.append(lst[i])
        else:
            fraud_data.append(lst[i])
    return non_fraud_data, fraud_data

# 
def concat_arrays(np_array1, np_array2, axis):
    return np.concatenate((np_array1, np_array2), axis=axis)


# 
def npTileArray(np_array, replicate_num):
    return np.tile(np_array,(replicate_num,1))

def npTileArrayOneDimension(np_array, replicate_num):
    return np.tile(np_array,replicate_num)