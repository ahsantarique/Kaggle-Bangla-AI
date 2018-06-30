import numpy as np


def toCategorical(y, nb_classes = 10):
    targets = y.reshape(-1)
    one_hot_targets = np.eye(nb_classes)[targets]
    return one_hot_targets



def main():
	y = np.loadtxt('y.txt', 'uint8')
	ycat = toCategorical(y=y)
    	np.savetxt('ycat.txt', ycat, fmt = '%1.0f')


if __name__ == '__main__':
	main()
