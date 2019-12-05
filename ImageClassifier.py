from __future__ import print_function
import numpy as np
import pickle as pkl
import os
import matplotlib.pyplot as plt
from keras.datasets import cifar100
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils
from keras.regularizers import l2


np.random.seed(2222)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

model_name = 'cifar100.h5'
validation = []
predictions = 20
batch_size = 60
num_classes = 100
num_epoch = 1
dropout_rate = 0.2
momentum_rate = 0.9
learning_rate = 0.1
decay_rate = 0.00005

# print and load the data provided by the cifar100 dataset.
(X_train, y_train), (X_test, y_test) = cifar100.load_data()
print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

# transform data into classes or categories. In this case our labels are Y and are images are X.
y_train = np_utils.to_categorical(y_train, num_classes)
y_test = np_utils.to_categorical(y_test, num_classes)

# load the data from CIFAR100.
# normalize the images in the data set.
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255.0
X_test /= 255.0

model = Sequential()
model.add(Conv2D(128, (3, 3), padding='same', input_shape=X_train.shape[1:]))
model.add(Activation('elu'))
model.add(Conv2D(128, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(256, (3, 3), padding='same'))
model.add(Activation('elu'))
model.add(Conv2D(256, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(dropout_rate))

model.add(Conv2D(512, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(512, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))

model.add(Flatten())
model.add(Dense(1024))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes))
model.add(Activation('softmax'))

model.summary()

opt = SGD(lr=learning_rate, momentum=momentum_rate, decay=decay_rate, nesterov=True)
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['acc'])

file = model_name
ckeckpoint = ModelCheckpoint(filepath=file, monitor='loss', verbose=1, save_best_only=True, mode='min')
callbacks = [ckeckpoint]

# training_history = model.fit(X_train, y_train, batch_size=batch_size, epochs=num_epoch, verbose=1,
#                              validation_data=(X_train, y_train),
#                              shuffle=True)

print('With data augmentation')
datagen = ImageDataGenerator(featurewise_center=False, samplewise_center=False,
                             featurewise_std_normalization=False, samplewise_std_normalization=False,
                             zca_whitening=False,
                             rotation_range=0,
                             width_shift_range=0.1,
                             height_shift_range=0.1,
                             horizontal_flip=True,
                             vertical_flip=False)

datagen.fit(X_train)

training_history = model.fit_generator(datagen.flow(X_train, y_train, batch_size=batch_size),
                                       steps_per_epoch=X_train.shape[0] // batch_size,
                                       epochs=num_epoch)

#
validation.append(model.evaluate_generator(datagen.flow(X_test, y_test, batch_size=batch_size),
                                           steps=X_test.shape[0] // batch_size, verbose=1))

pkl.dump(validation, open("loss_validation.p", 'wb'))
model.save_weights(model_name, overwrite=True)

# Load the label names to the prediction results to pass to the poetry generator for poem selection.
label_list_path = 'datasets/cifar-100-python/meta'

keras_dir = os.path.expanduser(os.path.join('~', '.keras'))
datadir_base = os.path.expanduser(keras_dir)
if not os.access(datadir_base, os.W_OK):
    datadir_base = os.path.join('/tmp', '.keras')
label_list_path = os.path.join(datadir_base, label_list_path)

with open(label_list_path, mode='rb') as f:
    labels = pkl.load(f)

# Evaluate model with test data set and share prediction results.
evaluation = model.evaluate_generator(datagen.flow(X_test, y_test,
                                      batch_size=batch_size),
                                      steps=X_test.shape[0] // batch_size)

print('Model Accuracy = %.2f' % (evaluation[1]))
predict_gen = model.predict_generator(datagen.flow(X_test, y_test,
                                      batch_size=batch_size),
                                      steps=X_test.shape[0] // batch_size)

for predict_index, predicted_y in enumerate(predict_gen):
    actual_label = labels['fine_label_names'][np.argmax(y_test[predict_index])]
    predicted_label = labels['fine_label_names'][np.argmax(predicted_y)]
    print('Actual Label = %s vs. Predicted Label = %s' % (actual_label,
                                                          predicted_label))
    if predict_index == predictions:
        break

score = model.evaluate(X_test, y_test, batch_size=batch_size, verbose=0)
print('Test score:', score[0])
print('Test accuracy:', score[1])

training_loss = training_history.history['loss']
accuracy = training_history.history['accuracy']
epoch_count = range(1, len(training_loss) + 1)
plt.plot(epoch_count, training_loss, 'r-')
plt.plot(epoch_count, accuracy, 'b-')
plt.legend(['Training Loss', 'Accuracy'])
plt.xlabel('Number of Epochs')
plt.ylabel('Accuracy or Loss')
plt.show()

