import numpy as np
import scipy.io

def dataReader(path,name):
    # path = 
    data = []
    print("Reading Data from file:", name)
    data = scipy.io.loadmat(path+"/"+name)[name]
    return np.array(data)

def labelClassExtract(train_array, index):
    list_index = []
    for i in range (train_array.shape[0]):
        list_index.append(train_array[i][index])
    # 
    return list_index

def predictAccuracy(set1, set2):
    intersection = []
    union = []
    cal_foreach = []
    for i in range (set1.shape[0]):
        for j in range (set1.shape[1]):
            # intersection
            if set1[i][j] == set2[i][j] and set1[i][j] == 1 and set2[i][j] == 1:
                intersection.append(set1[i][j])
            # union 
            if set1[i][j] == 1 or set2[i][j] == 1:
                union.append(1)

        cal_foreach.append(np.array(intersection).shape[0] / np.array(union).shape[0])
        # print(cal_foreach)
    return np.average(np.array(cal_foreach))