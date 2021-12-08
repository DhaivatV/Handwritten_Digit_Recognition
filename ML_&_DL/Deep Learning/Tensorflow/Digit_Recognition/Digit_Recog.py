from operator import le
from matplotlib.cm import get_cmap
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt
from tensorflow.python.ops.variables import trainable_variables
from sklearn.linear_model import LogisticRegression
from math import sqrt
from sklearn.metrics import mean_squared_error

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
  

learning_rate= 0.003
training_steps= 3000
batch_size= 250
display_step =100

n_hidden= 512


train_data = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_data= train_data.repeat().shuffle(60000).batch(batch_size).prefetch(1)

random_normal=tf.initializers.RandomNormal()
weights= {
             'h' : tf.Variable(random_normal([num_features, n_hidden])),
            'out': tf.Variable(random_normal([n_hidden, num_classes]))
        }
biases={
    'b': tf.Variable(tf.zeros([n_hidden])),
    'out':tf.Variable(tf.zeros([num_classes]))
}

def neural_net(input_data):
    hidden_layer = tf.add((tf.matmul(input_data,weights['h'])),biases['b'])
    hidden_layer= tf.nn.sigmoid(hidden_layer)

    out_layer = tf.matmul(hidden_layer, weights['out'])+biases['out']
    out_layer= tf.nn.softmax(out_layer)
    return tf.nn.softmax(out_layer)

def cross_entropy(y_pred, y_true):
    y_true= tf.one_hot(y_true, depth= num_classes)
    y_pred = tf.clip_by_value(y_pred, 1e-9, 1.)
    return tf.reduce_mean(-tf.reduce_sum(y_true*tf.math.log(y_pred)))

optimizer = tf.keras.optimizers.SGD(learning_rate)

def run_optimization(x, y):
    with tf.GradientTape() as g:
        pred= neural_net(x)
        loss= cross_entropy(pred, y)

    trainable_variables= list(weights.values()) + list(biases.values())

    gradients= g.gradient(loss, trainable_variables)

    return optimizer.apply_gradients(zip(gradients, trainable_variables))

def accu (y_pred, y_true):
    correct_prediction= tf.equal(tf.argmax(y_pred, 1), tf.cast(y_true, tf.int64))
    return tf.reduce_mean(tf.cast(correct_prediction, tf.float32), axis=-1)

for step, (batch_x, batch_y) in enumerate(train_data.take(training_steps),1):
    run_optimization(batch_x, batch_y)
    if step%display_step==0:
        pred=neural_net(batch_x)
        loss=cross_entropy(pred, batch_y)
        accuracy= accu(pred, batch_y)
        print("Training Epoch: %i, Loss: %f, Accuracy:%f"%(step, loss, accuracy))

pred= neural_net(x_test)
print("Test Accuracy: %f" % accu(pred, y_test   ))

n_images = 200
test_images = x_test[:n_images]
test_labels = y_test[:n_images]
predictions = neural_net(test_images)

for i in range(n_images):
    model_prediction = np.argmax(predictions.numpy()[i])
    if (model_prediction != test_labels[i]):
        plt.imshow(np.reshape(test_images[i], [28, 28]), cmap='gray_r')
        plt.title("Original Labels: %i,Model prediction: %i" % (test_labels[i],model_prediction))
        plt.show()
        # print("Original Labels: %i" % test_labels[i])
        # print("Model prediction: %i" % model_prediction)



# model= LogisticRegression()
# model.fit(x_train, y_train)

# pred= model.predict(x_test)



# print(sqrt(mean_squared_error(pred, y_test)))
# score= model.score(x_test, y_test)
# print(score)
