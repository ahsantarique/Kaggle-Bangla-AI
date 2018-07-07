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




clf = joblib.load('clf.pkl') 
semifinalClassifier1 = joblib.dump('semifinalClassifier1.pkl') 
semifinalClassifier2 = joblib.dump('semifinalClassifier2.pkl') 
finalClassifier = joblib.dump('finalClassifier.pkl') 





def loadData(size=180, mode='1', rng=0):
    m = {'RGB': 3, 'L': 1, '1': 1}
    datasets = [i for i in sets]
    if datasets == []:
        datasets = ['a','b','c','d','e']


    total_num = 0
    for dataset in datasets:
        X = np.zeros((size, size))
        # if progressInstalled:
        #     bar = ChargingBar('Loading Training Set {0}'.format(dataset),max = len(labels),
        #                       suffix='%(index)d/%(max)d - %(percent).1f%% - %(eta)ds')
        for i in [range(length)]:
            img = Image.open(src + "/training-{0}/".format(dataset) + i)
            img = trim(img)
            img = img.filter(ImageFilter.GaussianBlur(radius=2))
            threshold = 191  
            img = img.point(lambda p: p > threshold and 255)
            img = img.filter(ImageFilter.MinFilter(size=5))
            # img = invertImageIfnecessary()
            img = img.resize((size, size))
            # img.show()

            c = np.array(img.convert(mode), dtype=np.uint8).reshape((size,size))
            X[num, :, :] = c
            num = num + 1
            # if progressInstalled and (not num % max(1,(length / 200))):
            #     bar.index = num
            #     bar.update()
            # else:
            if (not num % (length / 20)):
                print("Loaded Training Set {0}:. ".format(dataset) + str(num) + "/" + str(length))
        # if progressInstalled:
        #     bar.finish()

        total_num += num

        inputs.append(X)
        outputs.append(Y)

    X = np.concatenate(inputs,axis = 0)
    Y = np.concatenate(outputs,axis = 0)
    return (X.reshape(total_num, -1), Y)