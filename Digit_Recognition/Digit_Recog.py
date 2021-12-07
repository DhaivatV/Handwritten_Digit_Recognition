from operator import le
from matplotlib.cm import get_cmap
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt
from tensorflow.python.ops.variables import trainable_variables
from sklearn.linear_model import LogisticRegression

num_classes = 10
num_features = 784
(x_train, y_train),(x_test, y_test) = mnist.load_data()
x_train , x_test = np.array(x_train, np.float32), np.array(x_test, np.float32)
x_train, x_test = x_train.reshape([-1, num_features]), x_test.reshape([-1, num_features])
x_train, x_test = x_train/255, x_test/255

def displayimgs (num):
    label = y_train[num]
    image = x_train[num].reshape([28, 28])
    plt.title("sample : %d, label : %d"%(num,label))
    plt.imshow(image, cmap= plt.get_cmap('gray_r'))  
    plt.show()

displayimgs(144)

clf = LogisticRegression()
clf.fit(x_train, y_train)

prediction = clf.predict(x_test[144].reshape([1, 784]))


    



