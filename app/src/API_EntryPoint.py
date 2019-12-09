import h5py
from keras.models import load_model
from keras.preprocessing.image import load_img
from scipy import misc
import tensorflow as tf
from flask import Flask, request
from skimage import io
import os
import pickle as pkl
import pandas as pd
import numpy as np

app = Flask(__name__)


def imageClassify(image_in_from_url, model):
    path_to_labels = 'dataset/cifar-100-python/meta'
    data_dir = os.path.expanduser(os.path.join('~', '.keras'))
    if not os.access(data_dir, os.W_OK):
        data_dir = os.path.join('/tmp', '.keras')
    path_to_labels = os.path.join(data_dir, path_to_labels)

    with open(path_to_labels, mode='rb') as f:
        labels = pkl.load(f)

    classify = model.predict_proba(np.reshape(image_in_from_url, (1, 32, 32, 3)), batch_size=1, verbose=0)
    prediction = pd.DataFrame(data=np.reshape(classify, 100), index=labels['fine_label_names'],
                              columns={'probability'}).sort_values('probability', ascending=False)
    prediction['name'] = prediction.index

    return prediction[:3]


# GET poem based off image in.
@app.route('/imageIn', methods=['GET', 'POST'])
def getPoemBasedOffImg():
    model_name = 'cifar100.h5'
    model = load_model(model_name)

    try:
        image = io.imread(request.form['image_url'])
        reshape_image = load_img(image, target_size=(224, 224))
        prediction = imageClassify(reshape_image, model)
        data = [{'name': x} for x in zip(prediction.iloc[:, 1], prediction.iloc[:, 0])]
    except:
        data = [{'name': 'Error'}]

    return data


if __name__ == '__main__':
    app.run(debug=True)
