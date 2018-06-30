# Import the modules
from sklearn.externals import joblib
from sklearn import datasets
from skimage.feature import hog
from sklearn.svm import LinearSVC, SVC
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score as accuracy
from sklearn.neural_network import MLPClassifier

def loadDataSet():
	features = np.loadtxt('X.txt', dtype=np.uint8)
	labels = np.loadtxt('y.txt', dtype=np.uint8)

	return features, labels


def getHogFeatures(features):
	list_hog_fd = []
	for feature in features:
	    fd = hog(feature.reshape((28, 28)), orientations=9, pixels_per_cell=(14, 14), cells_per_block=(1, 1), visualize=False)
	    list_hog_fd.append(fd)

	hog_features = np.array(list_hog_fd, 'float64')
	return hog_features




def getIndexFromCategorical(y):
    for i in range(len(y)):
        if(y[i] == 1):
            return i


def neuralNet():
    clf =  MLPClassifier(hidden_layer_sizes=(256,128,32,16), activation='relu', 
                    alpha=0.005, batch_size=64, early_stopping=True, 
                    learning_rate_init=0.001, solver='adam', learning_rate='adaptive', nesterovs_momentum=True, 
                    max_iter=1000, tol=1e-8, verbose=False, validation_fraction=0.1)

    clf.classes_ = 10

    return clf


def comparePrediction(y_true, y_pred):
    print("true", "prediction")
    for i in range(len(y_true)):
        print(y_true[i], y_pred[i])


def splitMetrics(clf, X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.9)

    clf.fit(X_train, y_train)
    y_test_pred = clf.predict(X_test)
    y_train_pred = clf.predict(X_train)

    comparePrediction(y_test, y_test_pred)
    # comparePrediction(y_train, y_train_pred)

    print(accuracy(y_test, y_test_pred));
    print(accuracy(y_train, y_train_pred));

    return



def main():
	features, labels = loadDataSet()

	n = len(labels)
	print(n)

	# hog_features = getHogFeatures(features)

	# print(hog_features.shape)


	clf = neuralNet()

	splitMetrics(clf, features, labels)
	print("done")


if __name__ == '__main__':
	main()