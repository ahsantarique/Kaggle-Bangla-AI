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
	clf =  MLPClassifier(hidden_layer_sizes=(512,256,64,16), activation='relu', 
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
	finalClassifier = finalNeuralNet()

	y_train_pred_using_individual_nn = np.empty((len(X_train),0), int)
	ycat_train = toCategorical(y_train)

	for i in range(10):
		clf[i].fit(X_train, ycat_train[:,i])
		print(y_train_pred_using_individual_nn.shape)
		print(np.array(clf[i].predict(X_train)).transpose().shape)
		y_train_pred_using_individual_nn = np.append(y_train_pred_using_individual_nn, np.array([clf[i].predict(X_train)]).transpose(), axis = 1)


	# print(y_train_pred_using_individual_nn)


	X_train_final = np.append(y_train_pred_using_individual_nn, X_train, axis = 1)

	finalClassifier.fit(X_train_final, y_train)

	np.savetxt('final_features_train.txt', y_train_pred_using_individual_nn, fmt = '%1.0f')
	# print(X_train_final.shape)
	# clf.fit(X_train, y_train)
	# y_test_pred = clf.predict(X_test)
	# comparePrediction(y_test, y_test_pred)
	# # comparePrediction(y_train, y_train_pred)
	# print(accuracy(y_test, y_test_pred));
	# print(accuracy(y_train, y_train_pred));
	print("training complete...")
	return clf, finalClassifier



def test(clf, finalClassifier, X_test):
	y_test_pred_using_individual_nn = np.empty((len(X_test),0), int)

	for i in range(10):
		print(y_test_pred_using_individual_nn.shape)
		print(np.array(clf[i].predict(X_test)).transpose().shape)
		y_test_pred_using_individual_nn = np.append(y_test_pred_using_individual_nn, np.array([clf[i].predict(X_test)]).transpose(), axis = 1)


	# print(y_train_pred_using_individual_nn)

	np.savetxt('final_features_test.txt', y_test_pred_using_individual_nn, fmt = '%1.0f')
	X_test_final = np.append(y_test_pred_using_individual_nn, X_test,  axis = 1)

	return finalClassifier.predict(X_test_final)





def ensemble(X, y):
	X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.9)
	clf, finalClassifier = train(X_train, y_train)
	y_test_pred = test(clf, finalClassifier, X_test)
	y_train_pred = test(clf, finalClassifier, X_train)
	comparePrediction(y_test_pred, y_test)
	# comparePrediction(y_train, y_train_pred)

	print(accuracy(y_test, y_test_pred));
	print(accuracy(y_train, y_train_pred));

	return



def bagging(X,y):
	seed = 7
	kfold = model_selection.KFold(n_splits=10, random_state=seed)
	cart = DecisionTreeClassifier()
	num_trees = 100
	model = BaggingClassifier(base_estimator=cart, n_estimators=num_trees, random_state=seed)
	results = model_selection.cross_val_score(model, X, y, cv=kfold)
	print(results.mean(), results.std())



def adaboost(X,y):
	seed = 8
	num_trees = 100
	kfold = model_selection.KFold(n_splits=10, random_state=seed)
	model = AdaBoostClassifier(n_estimators=num_trees, random_state=seed)
	results = model_selection.cross_val_score(model, X, y, cv=kfold)
	print(results.mean(), results.std())


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
