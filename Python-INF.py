#!/usr/bin/env python
# coding: utf-8

# ## INFS 772 Assignment 4 
# ### Total 10 points (6 tasks)
# ### Mohammad Shafayet Jamil Hossain

# # Image Classification with MNIST Dataset

# This short introduction uses [Keras](https://www.tensorflow.org/guide/keras/overview) to:
# 
# 1. Build a neural network that classifies images.
# 2. Train this neural network.
# 3. And, finally, evaluate the accuracy of the model.

# Download and install TensorFlow 2. Import TensorFlow into your program:
# 
# Note: Upgrade `pip` to install the TensorFlow 2 package. See the [install guide](https://www.tensorflow.org/install) for details.

# In[53]:


import tensorflow as tf

from tensorflow.keras.datasets import mnist #from Keras download the mnist dataset
from tensorflow.keras.models import Sequential # import the sequential model
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense # import the Dense(Fully connected) layer
from tensorflow.keras.optimizers import RMSprop # import rmsprop optimizer
from tensorflow.keras.utils import to_categorical # prepare the labels
from tensorflow.keras.layers import  Conv2D, Dropout,  MaxPooling2D
import matplotlib.pyplot as plt #import plot as plt


# In[54]:


tf.__version__


# Load and prepare the [MNIST dataset](http://yann.lecun.com/exdb/mnist/). Convert the samples from integers to floating-point numbers:

# In[55]:


mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0


# Build the `tf.keras.Sequential` model by stacking layers. Choose an optimizer and loss function for training:

# In[56]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
plt.figure()
plt.imshow(x_train[1])
plt.colorbar()
plt.grid(False)
plt.show()


# ## Task 1: Build Neural Network Using the following code samples (2 points)
# 
# ```python
# 
# model = Sequential() # make our model Sequential
# 
# model.add(Flatten(input_shape=(28, 28)))
# model.add(Dense(50, activation = "relu")) # add a hidden layer
# model.add(Dense(10, activation = 'softmax')) # add an output layer
# 
# ```
# ### Required Specification for the Hidden Layers
# 
# Dense 64 --> Dense 32
# 
# Total params: 52,650
# 

# In[57]:


model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(32, activation=tf.nn.relu),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])


# In[58]:


model.summary()


# The following cell is not required. Remove it if it doesn't work under your environment

# In[59]:


from IPython.display import SVG
from tensorflow.keras.utils import model_to_dot

SVG(model_to_dot(model, show_shapes= True, show_layer_names=True, dpi=65).create(prog='dot', format='svg'))


# ## Task 2: Test the Model Architecture (2 points)
# 
# All codes are provided and make sure they produce the expected outputs

# For each example the model returns a vector of "[logits](https://developers.google.com/machine-learning/glossary#logits)" or "[log-odds](https://developers.google.com/machine-learning/glossary#log-odds)" scores, one for each class.

# In[60]:


predictions = model(x_train[:1]).numpy()
predictions


# The `tf.nn.softmax` function converts these logits to "probabilities" for each class: 

# In[61]:


tf.nn.softmax(predictions).numpy()


# Note: It is possible to bake this `tf.nn.softmax` in as the activation function for the last layer of the network. While this can make the model output more directly interpretable, this approach is discouraged as it's impossible to
# provide an exact and numerically stable loss calculation for all models when using a softmax output. 

# The `losses.SparseCategoricalCrossentropy` loss takes a vector of logits and a `True` index and returns a scalar loss for each example.

# In[62]:


loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)


# This loss is equal to the negative log probability of the true class:
# It is zero if the model is sure of the correct class.
# 
# This untrained model gives probabilities close to random (1/10 for each class), so the initial loss should be close to `-tf.math.log(1/10) ~= 2.3`.

# In[63]:


loss_fn(y_train[:1], predictions).numpy()


# ## Task 3: Compile the Model (1 point)
# 
# ```python
# 
# # rmsprop optimizer
# model.compile(optimizer = "rmsprop", loss = loss_fn, metrics=['accuracy'])
# 
# # Adam optimizer
# model.compile(optimizer = "adam", loss = loss_fn, metrics=['accuracy'])
# ```
# 
# ### Requirement: compile either rmsprop or adam

# In[64]:


# your codes here
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


# ## Task 4: Fit the Model (1 point)
# 
# ### Requirement: 10 Epochs and save the outputs into history
# 
# The `Model.fit` method adjusts the model parameters to minimize the loss: 

# In[65]:


# your codes here
model.fit(x_train, y_train, epochs=5)
model.evaluate(x_test, y_test)


# The `Model.evaluate` method checks the models performance, usually on a "[Validation-set](https://developers.google.com/machine-learning/glossary#validation-set)" or "[Test-set](https://developers.google.com/machine-learning/glossary#test-set)".

# In[84]:


model.evaluate(x_test,  y_test, verbose=2)


# The image classifier is now trained to ~98% accuracy on this dataset. To learn more, read the [TensorFlow tutorials](https://www.tensorflow.org/tutorials/).

# If you want your model to return a probability, you can wrap the trained model, and attach the softmax to it:

# In[85]:


history = model.fit(x_test,  y_test, validation_split=0.33, epochs=150, batch_size=10, verbose=0)


# In[86]:


print(history.history.keys())


# ## Task 5: Visualize training loss and accuracy (1 point)
# 
# #### Codes are provided, make sure the outputs are displayed properly.
# 

# In[81]:


import matplotlib.pyplot as plt

# summarize history for accuracy
plt.plot(history.history['accuracy'])

plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])

plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


# ## Task 6: Build and train a simple CNN model and compare it with the Dense model (3 points)
# Which one is better and why (compare the testing accuracy)

# ### Build, train, and evaluate a simplle CNN model
# ### Reference: https://keras.io/examples/vision/mnist_convnet/ (2 points)
# 
# [Note] The CNN model architecture (and learning process configuration) can be idential to the one in the reference page.

# In[89]:


# keras imports for the dataset and building our neural network
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, MaxPool2D, Flatten
from keras.utils import np_utils

# to calculate accuracy
from sklearn.metrics import accuracy_score

# loading the dataset
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# building the input vector from the 28x28 pixels
X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

# normalizing the data to help with the training
X_train /= 255
X_test /= 255

# encoding using keras' numpy-related utilities
n_classes = 10
print("Shape before encoding: ", y_train.shape)
Y_train = np_utils.to_categorical(y_train, n_classes)
Y_test = np_utils.to_categorical(y_test, n_classes)
print("Shape after encoding: ", Y_train.shape)

# building a linear stack of layers with the sequential model
model = Sequential()
# convolutional layer
model.add(Conv2D(25, kernel_size=(3,3), strides=(1,1), padding='valid', activation='relu', input_shape=(28,28,1)))
model.add(MaxPool2D(pool_size=(1,1)))
# flatten output of conv
model.add(Flatten())
# hidden layer
model.add(Dense(32, activation='relu'))
# output layer
model.add(Dense(10, activation='softmax'))

# compiling the sequential model
model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')

# training the model for 10 epochs
model.fit(X_train, Y_train, batch_size=128, epochs=10, validation_data=(X_test, Y_test))
import matplotlib.pyplot as plt

# summarize history for accuracy
plt.plot(history.history['accuracy'])

plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])

plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


# ### Discuss what you learn from comparing the Dense model and the CNN model (1 point)

# Your answers here: The key distinction between the Convolutional Neural Network model and the Dense model, as previously stated, is that the Convolutional Neural Network layer utilizes fewer parameters by forcing input values to share the parameters. The Dense model employs a linear operation, which means that each output is created by the function depending on each input. Therefore the dense model is better over here.
