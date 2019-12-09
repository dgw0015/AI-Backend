from __future__ import print_function
import numpy as np
import pickle as pkl
import os
import matplotlib.pyplot as plt
import cv2
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

num_classes = 11
model_name = 'imageClassifier.h5'
categories = ["Dog", "Cat", "Mountain", "Beach", "Bird", "Forest", "Waterfall", "Tiger", "Flower", "Horse", "Castle"]

img_array =
new_array = cv2.resize()

train_datagen = ImageDataGenerator(rescale=1. / 255., preprocessing_function=preprocess_input)
train_generator = train_datagen.flow_from_directory('C:/Users/Drew Waller/Documents/AI-Backend/app/data/images/train',
                                                    target_size=(150, 150),
                                                    color_mode='rgb',
                                                    batch_size=25,
                                                    class_mode="categorical")

valid_datagen = ImageDataGenerator(rescale=1. / 255., preprocessing_function=preprocess_input)
valid_generator = valid_datagen.flow_from_directory('C:/Users/Drew Waller/Documents/AI-Backend/app/data/images/valid',
                                                    target_size=(150, 150),
                                                    color_mode='rgb',
                                                    batch_size=25,
                                                    class_mode="categorical")

model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=(150, 150, 3), activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3), activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3), activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(64, activation="relu"))
model.add(Dense(2, activation="softmax"))

model.summary()

opt = SGD(lr=0.01, momentum=0.9, decay=0.00005)
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
ckeckpoint = ModelCheckpoint(filepath=model_name, monitor='loss', verbose=1, save_best_only=True, mode='min')
callbacks = [ckeckpoint]

model.fit_generator(train_generator,
                    epochs=10,
                    validation_data=valid_generator)

base_model = VGG16(weights='imagenet', include_top=False, input_shape=(150, 150, 3))

base_model.summary()

tl_model = Sequential()
tl_model.add(base_model)
tl_model.add(Flatten())
tl_model.add(Dense(2, activation="softmax"))

tl_model.summary()

history = tl_model.fit_generator(train_generator,
                                 epochs=10,
                                 validation_data=valid_generator,
                                 callbacks=callbacks)

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
