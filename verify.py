#!/usr/bin/env python3

import argparse
import logging
import sys

import h5py
import numpy as np
import pandas as pd
import skimage.io
from keras import __version__ as keras_version
from keras.models import load_model

import data


def get_model_keras_version(model_file):
    f = h5py.File(model_file, mode='r')
    return f.attrs.get('keras_version')


def do_predictions(model_filename, datadirs, batch_size):
    model = load_model(model_filename)

    train, validation = data.preload_data(datadirs, valid_test_size=0.0)
    all_results = []
    for offset in range(0, len(train), batch_size):
        batch_samples = train[offset:offset + batch_size]

        images = np.empty((len(batch_samples),) + data.OUTPUT_SHAPE, np.float32)
        angles = np.empty((len(batch_samples),), np.float32)
        for idx, row in enumerate(batch_samples.itertuples()):
            images[idx] = skimage.io.imread(row[data.Indices.center_image+1])
            angles[idx] = row[data.Indices.steering+1]

        all_results.append(
            pd.Series(model.predict(images, batch_size=batch_size)[0], name='predicted',
                      index=range(offset, offset + batch_size)))
    all_results_df = pd.concat(all_results)
    return train.join(all_results_df)


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('datadir', type=str, nargs='+', help='Directories with data')
    parser.add_argument('--model', type=str, help='Model to load')
    parser.add_argument('--batchsize', type=int, default=32)

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)s %(message)s')

    logging.critical('Starting verification')
    logging.critical('Started as: %s', ' '.join(sys.argv))
    logging.critical('Data directories: %s', args.datadir)
    logging.critical('Model: %s', args.model)
    logging.critical('Batch size: %d', args.batchsize)

    batch_size = args.batchsize

    logging.info('Loading data')

    train, validation = data.preload_data(args.datadir, valid_test_size=0.0)

    logging.critical('Train set size: %d', len(train))
    logging.critical('Validation set size: %d', len(validation))

    model_keras_version = get_model_keras_version(args.model)
    keras_version_str = str(keras_version).encode('utf-8')
    if model_keras_version != keras_version_str:
        logging.info('You are using Keras version %s, but the model was built using %s', model_keras_version, keras_version_str)

    model = load_model(args.model)

    for offset in range(0, len(train), batch_size):
        batch_samples = train[offset:offset + batch_size]

        images = np.empty((batch_size,) + data.OUTPUT_SHAPE, np.float32)
        angles = np.empty((batch_size,), np.float32)
        for idx, row in enumerate(batch_samples.itertuples()):
            images[idx] = skimage.io.imread(row[data.Indices.center_image+1])
            angles[idx] = row[data.Indices.steering+1]

        result = pd.Series(model.predict(images, batch_size=batch_size)[0], name='predicted',
                           index=range(offset, offset + batch_size))

        #print(pd.Series(angles-result[0], name='delta'))

if __name__ == '__main__':
    main()
