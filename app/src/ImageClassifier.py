from __future__ import print_function
import numpy as np
import pickle as pkl
import os
import matplotlib.pyplot as plt
from keras.applications import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

train_datagen=ImageDataGenerator(rescale=1./255., preprocessing_function=preprocess_input) #included in our dependencies

train_generator=train_datagen.flow_from_directory('dog vs cat/dataset/train',
                                                 target_size=(150,150),
                                                 color_mode='rgb',
                                                 batch_size=16,
                                                 class_mode='categorical')


num_classes = 11
model_name = 'imageClassifier.h5'

model = Sequential()

model.add(Conv2D(128, (3, 3), padding='same'))
model.add(Activation('elu'))
model.add(Conv2D(128, (3, 3)))
model.add(Activation('elu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(256, (3, 3), padding='same'))
model.add(Activation('elu'))
model.add(Conv2D(256, (3, 3)))
model.add(Activation('elu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512))
model.add(Activation('elu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes))
model.add(Activation('softmax'))

model.summary()

opt = SGD(lr=learning_rate, momentum=momentum_rate, decay=decay_rate)
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
ckeckpoint = ModelCheckpoint(filepath=model_name, monitor='loss', verbose=1, save_best_only=True, mode='min')
callbacks = [ckeckpoint]

# plot loss during training
plt.title('Loss')
plt.plot(history.history['loss'], 'b', label='train')
plt.plot(history.history['val_loss'], 'r', label='test')
plt.legend()
plt.show()

# plot accuracy during training
plt.title('Accuracy')
plt.plot(history.history['accuracy'], 'b', label='train')
plt.plot(history.history['val_accuracy'], 'g', label='test')
plt.legend()
plt.show()




