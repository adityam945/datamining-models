
from DataProcessor import *
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from SVM import *

def main():
# Read data
    # train data and labels
    X_train = dataReader("./data", 'X_train')
    y_train = dataReader("./data", 'y_train')
    # test data and labels
    X_test = dataReader("./data", 'X_test')
    y_test = dataReader("./data", 'y_test')

    # poly
    predicted_poly = np.empty((0, 1), int)
    for i in range(0, 6):
        SVM_each = CalSVM(X_train, y_train, X_test, y_test, i, "poly")
        if predicted_poly.shape[0] == 0:
            predicted_poly = SVM_each
        else:
            predicted_poly = np.append(predicted_poly, SVM_each, axis=1)

    predicted_poly_accuracy = predictAccuracy(y_test, predicted_poly)
    print("Polynomial SVM", predicted_poly_accuracy)
    # gauss
    predicted_gauss = np.empty((0, 1), int)
    for i in range(0, 6):
        SVM_each = CalSVM(X_train, y_train, X_test, y_test, i, "rbf")
        if predicted_gauss.shape[0] == 0:
            predicted_gauss = SVM_each
        else:
            predicted_gauss = np.append(predicted_gauss, SVM_each, axis=1)
    
    predicted_gauss_accuracy = predictAccuracy(y_test, predicted_gauss)
    print("Gaussian SVM", np.average(predicted_gauss_accuracy))


main()