#!/usr/bin/env python3

import argparse
import logging
import sys

import h5py
import keras
import numpy as np
import pandas as pd
import skimage.io
import cv2
from keras import __version__ as keras_version
from keras.layers import Lambda, K
from keras.models import load_model, Sequential

import data


def get_model_keras_version(model_file):
    f = h5py.File(model_file, mode='r')
    return f.attrs.get('keras_version')


def get_model(model_filename):
    model = load_model(model_filename, custom_objects={'rmse': data.rmse})

    model_keras_version = get_model_keras_version(model_filename)
    keras_version_str = str(keras_version).encode('utf-8')
    if model_keras_version != keras_version_str:
        logging.info('You are using Keras version %s, but the model was built using %s', model_keras_version, keras_version_str)

    return model

def normalize(x):
    # utility function to normalize a tensor by its L2 norm
    return x / (K.sqrt(K.mean(K.square(x))) + 1e-5)

def grad_cam(input_model, image, category_index, layer_name):
    model = Sequential()
    model.add(input_model)

    nb_classes = 1000
    target_layer = lambda x: target_category_loss(x, category_index, nb_classes)
    model.add(Lambda(target_layer,
                     output_shape = target_category_loss_output_shape))

    loss = K.sum(model.layers[-1].output)
    conv_output =  [l for l in model.layers[0].layers if l.name is layer_name][0].output
    grads = normalize(K.gradients(loss, conv_output)[0])
    gradient_function = K.function([model.layers[0].input], [conv_output, grads])

    output, grads_val = gradient_function([image])
    output, grads_val = output[0, :], grads_val[0, :, :, :]

    weights = np.mean(grads_val, axis = (0, 1))
    cam = np.ones(output.shape[0 : 2], dtype = np.float32)

    for i, w in enumerate(weights):
        cam += w * output[:, :, i]

    w, h, _ = image.shape
    cam = cv2.resize(cam, (w, h))
    cam = np.maximum(cam, 0)
    heatmap = cam / np.max(cam)

    # Return to BGR [0..255] from the preprocessed image
    image = image[0, :]
    image -= np.min(image)
    image = np.minimum(image, 255)

    cam = cv2.applyColorMap(np.uint8(255*heatmap), cv2.COLORMAP_JET)
    cam = np.float32(cam) + np.float32(image)
    cam = 255 * cam / np.max(cam)
    return np.uint8(cam), heatmap

def do_predictions(model, image_filename):
    img = skimage.io.imread(image_filename)

    w, h, _ = img.shape

    class_weights = model.layers[-1].get_weights()[0]


    return train.join(all_results_df)


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('image', type=str, help='Image to load')
    parser.add_argument('--model', type=str, help='Model to load')

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)s %(message)s')

    logging.critical('Starting verification')
    logging.critical('Started as: %s', ' '.join(sys.argv))
    logging.critical('Data directories: %s', args.datadir)
    logging.critical('Model: %s', args.model)



if __name__ == '__main__':
    main()
