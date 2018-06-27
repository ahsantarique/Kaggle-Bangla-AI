import pandas as pd
import numpy as np
import sys
from PIL import Image, ImageChops
import matplotlib.pyplot as plt

# progressInstalled = False
# try:
#     from progress.bar import ChargingBar
# except:
#     progressInstalled = False

sets = set(['a','b','c','d','e'])
src = "/home/ahsan/Desktop/kaggle bangla ai/numta"

MAX_LABELS = 1



def trim(im):
    bg = Image.new(im.mode, im.size, im.getpixel((0,0)))
    diff = ImageChops.difference(im, bg)
    diff = ImageChops.add(diff, diff, 2.0, -100)
    bbox = diff.getbbox()
    if bbox:
        return im.crop(bbox)




def loadData(size=180, mode='L', rng=0):
    m = {'RGB': 3, 'L': 1, '1': 1}
    inputs = []
    outputs = []
    datasets = [i for i in sets]
    if datasets == []:
        datasets = ['a','b','c','d','e']

    for dataset in datasets:
        labels = pd.read_csv(src + "/training-{0}.csv".format(dataset))
        if rng == 0:
            labels = labels[['filename', 'digit']]
        else:
            labels = labels[['filename', 'digit']][rng[0]:rng[1]]

        length = min(len(labels), MAX_LABELS)

        Y = np.array(labels['digit'][range(length)],dtype=np.uint8)

        X = np.zeros((length, size, size))
        num = 0
        # if progressInstalled:
        #     bar = ChargingBar('Loading Training Set {0}'.format(dataset),max = len(labels),
        #                       suffix='%(index)d/%(max)d - %(percent).1f%% - %(eta)ds')
        for i in labels['filename'][range(length)]:
            img = Image.open(src + "/training-{0}/".format(dataset) + i)
            img = trim(img)
            img.show()
            c = img.resize((size, size))
            c = np.array(c.convert(mode), dtype=np.uint8).reshape((size,size))
            X[num, :, :] = c
            num = num + 1
            # if progressInstalled and (not num % max(1,(length / 200))):
            #     bar.index = num
            #     bar.update()
            # else:
            #     if (not num % (length / 20)):
            #         print("Loaded Training Set {0}:. ".format(dataset) + str(num) + "/" + str(length))
            if (not num % (length / 20)):
                print("Loaded Training Set {0}:. ".format(dataset) + str(num) + "/" + str(length))
        # if progressInstalled:
        #     bar.finish()
        inputs.append(X)
        outputs.append(Y)

    X = np.concatenate(inputs,axis = 0)
    Y = np.concatenate(outputs,axis = 0)
    return (X, Y)


def show_digit(x,label):
    plt.axis('off')
    l = x.shape[1]
    m = x.shape[2]
    if m == 1:
        plt.imshow(x.reshape((l,l)), cmap=plt.cm.gray)
    else:
        plt.imshow(x.reshape((l, l, m)))
    #plt.title(label)
    plt.show()
    return


def vis_image(index, X, Y):
    label = Y[index]
    show_digit(X[index, :, :, :],label)
    print("Label " + str(label))
    return


def main():
    X,y = loadData();
    print(X)
    print(y)
    np.savetxt('X.txt', X)
    np.savetxt('y.txt', y)



if __name__ == '__main__':
    main()