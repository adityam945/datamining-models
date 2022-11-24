from sklearn.svm import SVC
from DataProcessor import *

def CalSVM(X_train, y_train, X_test, y_test, index, kernel_type):

    y_train_index = np.array(labelClassExtract(y_train, index))
    y_test_index = np.array(labelClassExtract(y_test, index))

    SVM = SVC(kernel = kernel_type , degree = 2)
    SVM.fit(X_train, y_train_index)
    svm_predictions = SVM.predict(X_test)
    svm_predictions = np.array(svm_predictions)
    # accuracy = predictAccuracy(svm_predictions, y_test_index)
    svm_predictions_axis = np.empty((0, 1), int)
    for i in range (0, svm_predictions.shape[0]):
        svm_predictions_axis = np.append(svm_predictions_axis, np.array([[svm_predictions[i]]]), axis=0)

    return np.array(svm_predictions_axis)