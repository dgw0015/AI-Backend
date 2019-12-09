from __future__ import print_function
import numpy as np
import pickle as pkl
import os
import matplotlib.pyplot as plt
from keras.datasets import cifar100
from keras.applications import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import SGD, Adam
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

num_classes = 100
batch_size = 100
num_epoch = 10
validation = []
predictions = 40
use_data_augmentation = False
model_name = 'cifar100.h5'

# print and load the data provided by the cifar100 dataset.
(train_images, train_labels), (test_images, test_labels) = cifar100.load_data(label_mode='fine')

print('train_images shape:', train_images.shape)
print(train_images.shape[0], 'train samples')
print(test_images.shape[0], 'test samples')

# transform data into classes or categories. In this case our labels are Y and are images are X.
train_labels = np_utils.to_categorical(train_labels, num_classes)
test_labels = np_utils.to_categorical(test_labels, num_classes)

train_images = train_images.astype('float32')
test_images = test_images.astype('float32')
train_images = train_images/255.0
test_images = test_images/255.0

# categories = ["beaches", "birds", "castles", "cats", "dogs", "flowers", "forests", "horses", "mountains", "tigers",
#               "waterfalls"]

classes = [
    'apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle',
    'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel',
    'can', 'castle', 'caterpillar', 'cattle', 'chair', 'chimpanzee', 'clock',
    'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur',
    'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster',
    'house', 'kangaroo', 'keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion',
    'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain', 'mouse',
    'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear',
    'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy', 'porcupine',
    'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose',
    'sea', 'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake',
    'spider', 'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table',
    'tank', 'telephone', 'television', 'tiger', 'tractor', 'train', 'trout',
    'tulip', 'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman',
    'worm'
]

model = Sequential()
model.add(Conv2D(64, (3, 3), padding='same', input_shape=train_images.shape[1:]))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3), padding='same'))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512))
model.add(Activation("relu"))
model.add(Dropout(0.5))
model.add(Dense(num_classes))
model.add(Activation("softmax"))

model.summary()

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
ckeckpoint = ModelCheckpoint(filepath=model_name, monitor='loss', verbose=1, save_best_only=True, mode='min')
callbacks = [ckeckpoint]

history = model.fit(train_images, train_labels,
                    batch_size=100, epochs=num_epoch, callbacks=callbacks, verbose=1)

validation.append(model.evaluate(test_images, test_labels, batch_size=batch_size, verbose=1))

# base_model = VGG16(weights='imagenet', include_top=False, input_shape=(150, 150, 3))
#
# base_model.summary()
#
# tl_model = Sequential()
# tl_model.add(base_model)
# tl_model.add(Flatten())
# tl_model.add(Dense(2))
# tl_model.add(Activation("softmax"))
#
# tl_model.summary()

# history = tl_model.fit_generator(train_generator,
#                                  epochs=10,
#                                  validation_data=valid_generator,
#                                  callbacks=callbacks)

# Data augmentation
if use_data_augmentation:
    datagen = ImageDataGenerator(featurewise_center=False, samplewise_center=False,
                                 featurewise_std_normalization=False, samplewise_std_normalization=False,
                                 zca_whitening=False,
                                 rotation_range=0,
                                 width_shift_range=0.1,
                                 height_shift_range=0.1,
                                 horizontal_flip=True,
                                 vertical_flip=False)

    datagen.fit(train_images)

    history = model.fit_generator(datagen.flow(train_images, train_labels, batch_size=batch_size),
                                  validation_data=(test_images, test_labels),
                                  steps_per_epoch=train_images.shape[0] // batch_size,
                                  epochs=num_epoch, callbacks=callbacks)

    validation.append(model.evaluate_generator(datagen.flow(test_images, test_labels, batch_size=batch_size),
                                               steps=test_images.shape[0] // batch_size,
                                               verbose=0))

pkl.dump(validation, open("loss_validation.p", 'wb'))

# Load the label names to the prediction results to pass to the poetry generator for poem selection.
label_list_path = 'datasets/cifar-100-python/meta'
keras_dir = os.path.expanduser(os.path.join('~', '.keras'))
datadir_base = os.path.expanduser(keras_dir)

if not os.access(datadir_base, os.W_OK):
    datadir_base = os.path.join('/tmp', '.keras')

label_list_path = os.path.join(datadir_base, label_list_path)  #
with open(label_list_path, mode='rb') as f:
    labels = pkl.load(f)

# Evaluate model with test data set and share prediction results.
print('\nStarting model evaluation.')
train_acc = model.evaluate(train_images, train_labels, batch_size=batch_size, verbose=1)

test_acc = model.evaluate(test_images, test_labels, batch_size=batch_size, verbose=1)

# Prints the models current accuracy after all training epochs.
print('Models Current Accuracy = %.2f' % (test_acc[1]))
predict_gen = model.predict(test_images, test_labels, batch_size=batch_size, verbose=1)

for predict_index, predicted_y in enumerate(predict_gen):
    actual_label = labels['fine_label_names'][np.argmax(test_labels[predict_index])]
    predicted_label = labels['fine_label_names'][np.argmax(predicted_y)]
    print('Actual = %s vs. Model Prediction = %s' % (actual_label, predicted_label))
    if predict_index == predictions:
        break

# print(train_images[0])
plt.imshow(train_images[0])
plt.show()

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
