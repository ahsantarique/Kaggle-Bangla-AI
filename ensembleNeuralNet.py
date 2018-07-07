# Import the modules
from sklearn.externals import joblib
from sklearn import datasets
from skimage.feature import hog
from sklearn.svm import LinearSVC, SVC
import numpy as np
from sklearn import model_selection
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score as accuracy
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier


def loadDataSet():
	X = np.loadtxt('X.txt', dtype=np.uint8)
	y = np.loadtxt('y.txt', dtype=np.uint8)

	return X, y


def getHogFeatures(features):
	list_hog_fd = []
	for feature in features:
	    fd = hog(feature.reshape((28, 28)), orientations=9, pixels_per_cell=(14, 14), cells_per_block=(1, 1), visualize=False)
	    list_hog_fd.append(fd)

	hog_features = np.array(list_hog_fd, 'float64')
	return hog_features


def toCategorical(y, nb_classes = 10):
    targets = y.reshape(-1)
    one_hot_targets = np.eye(nb_classes)[targets]
    return one_hot_targets



def getIndexFromCategorical(y):
	for i in range(len(y)):
		if(y[i] == 1):
			return i


def individualNeuralNet():
	clf =  MLPClassifier(hidden_layer_sizes=(256,128,32,16), activation='relu', 
		    alpha=0.005, batch_size=64, early_stopping=True, 
		    learning_rate_init=0.01, solver='adam', learning_rate='adaptive', nesterovs_momentum=True, 
		    max_iter=500, tol=1e-8, verbose=False, validation_fraction=0.1)

	return clf


def finalNeuralNet():
	clf =  MLPClassifier(hidden_layer_sizes=(32,16,8), activation='relu', 
		    alpha=0.001, batch_size=64, early_stopping=True, 
		    learning_rate_init=0.001, solver='adam', learning_rate='adaptive', nesterovs_momentum=True, 
		    max_iter=1000, tol=1e-8, verbose=False, validation_fraction=0.1)

	return clf


def semifinalNeuralNet1():
	clf =  MLPClassifier(hidden_layer_sizes=(16,8), activation='relu', 
		    alpha=0.001, batch_size=64, early_stopping=True, 
		    learning_rate_init=0.001, solver='adam', learning_rate='adaptive', nesterovs_momentum=True, 
		    max_iter=500, tol=1e-8, verbose=False, validation_fraction=0.1)

	return clf


def semifinalNeuralNet2():
	clf =  MLPClassifier(hidden_layer_sizes=(256,128,32,16), activation='relu', 
		    alpha=0.001, batch_size=64, early_stopping=True, 
		    learning_rate_init=0.001, solver='adam', learning_rate='adaptive', nesterovs_momentum=True, 
		    max_iter=1500, tol=1e-8, verbose=False, validation_fraction=0.1)

	return clf



def comparePrediction(y_true, y_pred):
	print("true", "prediction")
	for i in range(len(y_true)):
		print(y_true[i], y_pred[i])



def train(X_train, y_train):
	clf = [individualNeuralNet() for i in range(10)]
	semifinalClassifier1 = semifinalNeuralNet1()
	semifinalClassifier2 = semifinalNeuralNet2()
	finalClassifier = finalNeuralNet()


	y_train_pred_using_individual_nn = np.empty((len(X_train),0), int)
	ycat_train = toCategorical(y_train)

	for i in range(10):
		clf[i].fit(X_train, ycat_train[:,i])
		print(y_train_pred_using_individual_nn.shape)
		print(np.array(clf[i].predict(X_train)).transpose().shape)
		y_train_pred_using_individual_nn = np.append(y_train_pred_using_individual_nn, np.array([clf[i].predict(X_train)]).transpose(), axis = 1)


	# print(y_train_pred_using_individual_nn)


	# X_train_final = np.append(y_train_pred_using_individual_nn, X_train, axis = 1)

	semifinalClassifier1.fit(y_train_pred_using_individual_nn, y_train)

	semifinalClassifier2.fit(X_train, y_train)

	semifeature = np.empty((len(X_train),0), int)
	y1 = semifinalClassifier1.predict(y_train_pred_using_individual_nn)
	y2 = semifinalClassifier2.predict(X_train)


	y1 = toCategorical(y1)
	y2 = toCategorical(y2);

	# print(np.array([y1[:,0]]).shape, np.array([y2[:,0]]).shape)

	for i in range(10):
		semifeature = np.append(semifeature, np.array([y1[:,i]]).transpose(), axis = 1)
		semifeature = np.append(semifeature, np.array([y2[:,i]]).transpose(), axis = 1)


	finalClassifier.fit(semifeature, y_train)

	np.savetxt('final_features_train.txt', y_train_pred_using_individual_nn, fmt = '%1.0f')
	# print(X_train_final.shape)
	# clf.fit(X_train, y_train)
	# y_test_pred = clf.predict(X_test)
	# comparePrediction(y_test, y_test_pred)
	# # comparePrediction(y_train, y_train_pred)
	# print(accuracy(y_test, y_test_pred));
	# print(accuracy(y_train, y_train_pred));
	print("training complete...")
	return clf, semifinalClassifier1, semifinalClassifier2, finalClassifier



def test(clf, semifinalClassifier1, semifinalClassifier2, finalClassifier, X_test):
	y_test_pred_using_individual_nn = np.empty((len(X_test),0), int)

	for i in range(10):
		y_test_pred_using_individual_nn = np.append(y_test_pred_using_individual_nn, np.array([clf[i].predict(X_test)]).transpose(), axis = 1)



	semifeature = np.empty((len(X_test),0), int)
	y1 = semifinalClassifier1.predict(y_test_pred_using_individual_nn)
	y2 = semifinalClassifier2.predict(X_test)


	y1 = toCategorical(y1)
	y2 = toCategorical(y2);

	# print(np.array([y1[:,0]]).shape, np.array([y2[:,0]]).shape)

	for i in range(10):
		semifeature = np.append(semifeature, np.array([y1[:,i]]).transpose(), axis = 1)
		semifeature = np.append(semifeature, np.array([y2[:,i]]).transpose(), axis = 1)
	# print(y_train_pred_using_individual_nn)

	np.savetxt('final_features_test.txt', y_test_pred_using_individual_nn, fmt = '%1.0f')
	# X_test_final = np.append(y_test_pred_using_individual_nn, X_test,  axis = 1)

	return finalClassifier.predict(semifeature);





def ensemble(X, y):
	X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.99)
	clf, semifinalClassifier1, semifinalClassifier2, finalClassifier = train(X_train, y_train)
	y_test_pred = test(clf, semifinalClassifier1, semifinalClassifier2, finalClassifier, X_test)
	# y_train_pred = test(clf, semifinalClassifier1, semifinalClassifier2, finalClassifier, X_train)
	comparePrediction(y_test_pred, y_test)
	# comparePrediction(y_train, y_train_pred)

	print(accuracy(y_test, y_test_pred));
	# print(accuracy(y_train, y_train_pred));


	joblib.dump(clf, 'clf.pkl') 
	joblib.dump(semifinalClassifier1, 'semifinalClassifier1.pkl') 
	joblib.dump(semifinalClassifier2, 'semifinalClassifier2.pkl') 
	joblib.dump(finalClassifier, 'finalClassifier.pkl') 

	return






def main():
	X, y = loadDataSet()

	n = len(y)
	print(n)

	# hog_features = getHogFeatures(features)

	# print(hog_features.shape)

	ensemble(X, y)
	# bagging(X, y)
	# adaboost(X,y)

	print("done")


if __name__ == '__main__':
	main()
