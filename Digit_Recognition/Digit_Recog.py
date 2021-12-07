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


    

learning_rate= 0.0003
training_steps= 5000
batch_size= 300
display_step =300

n_hidden= 520


train_data = tf.data.Dataset.from_tensor_slices(x_train, y_train)
train_data= train_data.repeat().shuffle(60000).batch(batch_size).prefetch(1)

random_normal=tf.initializers.RandomNormal()
weights= {
             'h' : tf.Variable(random_normal(num_features, n_hidden)),
            'out': tf.Variable(random_normal(n_hidden, num_classes))
        }
biases={
    'b': tf.Variable(tf.zeros(n_hidden)),
    'out':tf.Variable(tf.zeros(num_classes))
}


