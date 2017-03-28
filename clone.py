#!/usr/bin/env python3

import argparse
import logging
import os
import random
import sys
import socket
from datetime import datetime

import keras.regularizers
import numpy as np
import skimage.io
import sklearn
import sklearn.utils
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.callbacks import TensorBoard
from keras.layers import Conv2D, MaxPooling2D, Cropping2D
from keras.layers import Flatten, Dense, Lambda, Dropout
from keras.models import Sequential

import data


def create_model_lenet(parameters):
    model = Sequential()
    model.add(Cropping2D(cropping=((50, 20), (0, 0)), input_shape=(160, 320, 3)))
    model.add(Lambda(lambda x: x / 255.0 - 0.5))
    model.add(Conv2D(6, (5, 5), activation='relu'))
    model.add(MaxPooling2D())
    model.add(Conv2D(6, (5, 5), activation='relu'))
    model.add(Flatten())
    model.add(Dense(120))
    model.add(Dense(84))
    model.add(Dense(1))
    return model


def create_model_nvidia(parameters):
    dropout = parameters.get('dropout', 0)
    l2_regularizer_lambda = parameters.get('l2_regularizer', 0)

    if l2_regularizer_lambda:
        l2_regularizer = keras.regularizers.l2(l2_regularizer_lambda)
    else:
        l2_regularizer = None

    model = Sequential()
    model.add(Cropping2D(cropping=((50, 20), (0, 0)), input_shape=(160, 320, 3)))
    model.add(Lambda(lambda x: x / 255.0 - 0.5))
    model.add(Conv2D(24, (5, 5), strides=(2, 2), activation='relu'))
    model.add(Conv2D(36, (5, 5), strides=(2, 2), activation='relu'))
    model.add(Conv2D(48, (5, 5), strides=(2, 2), activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(Flatten())

    if dropout:
        model.add(Dropout(rate=dropout))
    model.add(Dense(100, kernel_regularizer=l2_regularizer))
    if dropout:
        model.add(Dropout(rate=dropout))
    model.add(Dense(50, kernel_regularizer=l2_regularizer))
    if dropout:
        model.add(Dropout(rate=dropout))
    model.add(Dense(10, kernel_regularizer=l2_regularizer))
    if dropout:
        model.add(Dropout(rate=dropout))
    model.add(Dense(1, kernel_regularizer=l2_regularizer))
    return model


MODELS = {
    'lenet': create_model_lenet,
    'nvidia': create_model_nvidia
}

DEFAULT_MODEL = 'nvidia'


def create_model(name, parameters):
    logging.info('Creating model %s, parameters=%s', name, parameters)
    return MODELS[name](parameters)


def train_model(model,
                train_generator,
                validation_data,
                train_steps_per_epoch,
                validation_steps_per_epoch,
                initial_epoch,
                epochs,
                tf_logs_dir,
                checkpoints_dir,
                name,
                learning_rate):

    optimizer = keras.optimizers.Adam(lr=learning_rate)
    loss_function = 'mse'

    logging.info(('Training model {}: epochs={}, initial_epoch={}, train_steps_per_epoch={}, '
                  'validation_steps_per_epoch={}, loss={}, optimizer={}').format(name, epochs, initial_epoch,
                                                                                 train_steps_per_epoch,
                                                                                 validation_steps_per_epoch,
                                                                                 loss_function, optimizer))

    tensorboard_callback = TensorBoard(log_dir=tf_logs_dir, histogram_freq=1,
                                       write_graph=True, write_images=True)

    filepath = os.path.join(checkpoints_dir,
                            'model-' + name + '-improved-{epoch:02d}-{val_loss:.3f}.h5')

    checkpoint_callback = ModelCheckpoint(filepath, monitor='val_loss',
                                          verbose=0, save_weights_only=False,
                                          save_best_only=True)

    early_stopping_callback = EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=10, verbose=1)

    model.compile(loss=loss_function, optimizer=optimizer)
    model.fit_generator(train_generator,
                        steps_per_epoch=train_steps_per_epoch,
                        epochs=epochs,
                        initial_epoch=initial_epoch,
                        validation_data=validation_data,
                        validation_steps=validation_steps_per_epoch,
                        callbacks=[tensorboard_callback, checkpoint_callback, early_stopping_callback])


def generate_data(samples, batch_size=32, angle_adj=0.1):
    def gen_data(a_batch_samples, filename_idx, an_angle_adj):
        images = np.empty((len(a_batch_samples),) + data.OUTPUT_SHAPE, np.float32)
        angles = np.empty((len(a_batch_samples),), np.float32)

        for idx, row in enumerate(a_batch_samples.itertuples()):
            images[idx] = skimage.io.imread(row[filename_idx])
            angles[idx] = row[data.Indices.steering + 1] + an_angle_adj

        yield sklearn.utils.shuffle(images, angles)

    while True:
        sklearn.utils.shuffle(samples)

        for offset in range(0, len(samples), batch_size):
            batch_samples = samples[offset:offset + batch_size]
            yield from gen_data(batch_samples, data.Indices.center_image + 1, 0)

            if angle_adj is not None:
                yield from gen_data(batch_samples, data.Indices.left_image + 1, angle_adj)
                yield from gen_data(batch_samples, data.Indices.right_image + 1, -angle_adj)


def flip_images(batch_generator, flip_prob=0.5):
    for batch_images, batch_angles in batch_generator:
        images = batch_images.copy()
        angles = batch_angles.copy()
        indices = random.sample(range(0, len(batch_images)), int(len(batch_images) * flip_prob))
        for idx in indices:
            images[idx] = np.fliplr(images[idx])
            angles[idx] = -angles[idx]
        yield images, angles


def parse_arguments():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs to learn')
    parser.add_argument('--learning-rate', type=float, default=0.0001, help='Learning rate')
    parser.add_argument('--name', type=str, default='noname', help='Model name')
    default_out_dir = os.path.join('out', datetime.now().isoformat() + '.' + socket.getfqdn())
    parser.add_argument('--dest', type=str, default=default_out_dir, help='Destination directory')

    subparsers = parser.add_subparsers(dest='mode', help='Select one of the modes')

    new_model_parser = subparsers.add_parser('new', help='Start training new model',
                                             formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    new_model_parser.add_argument('--model', type=str, choices=MODELS.keys(), default=DEFAULT_MODEL, help='Model type')
    new_model_parser.add_argument('--model-dropout', type=float, default=0.0,
                                  help='Dropout (0.0 to disable, 1.0 - drop everything)')
    new_model_parser.add_argument('--model-l2-regularizer', type=float, default=0.0,
                                  help='L2 weights regularizer for FC layers')

    cont_model_parser = subparsers.add_parser('continue', help='Continue training exiting model',
                                              formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    cont_model_parser.add_argument('--model-file', type=str, required=True,
                                   help='Filename with model in .h5 keras format')
    cont_model_parser.add_argument('--initialepoch', type=int, default=0, help='Epoch to start counting from')

    data_group = parser.add_argument_group('data_parameters', 'Input data processing parameters')
    data_group.add_argument('--batchsize', type=int, default=32)
    data_group.add_argument('--angleadj', type=float, default=None, help='Angle adjustment for left-right images')
    data_group.add_argument('--validsize', type=float, default=0.2, help='Validation test size')

    new_model_parser.add_argument('datadir', type=str, nargs='+', help='Directories with data')
    cont_model_parser.add_argument('datadir', type=str, nargs='+', help='Directories with data')

    args = parser.parse_args()
    return args


def main():
    args = parse_arguments()

    logging_filename = os.path.join(args.dest, 'clone.log')
    os.makedirs(os.path.dirname(logging_filename), exist_ok=True)

    logging.basicConfig(filename=logging_filename,
                        level=logging.INFO,
                        format='%(asctime)s %(levelname)s %(message)s')

    logging.critical('Starting cloning')
    logging.critical('Started as: %s', ' '.join(sys.argv))
    logging.critical('Data directories: %s', args.datadir)
    logging.critical('Batch size: %d', args.batchsize)
    logging.critical('Learning rate: %s', args.learning_rate)
    logging.critical('Epochs: %d', args.epochs)
    logging.critical('Angle adjustment: %s', args.angleadj)
    logging.critical('Validation set size: %s', args.validsize)
    logging.critical('Model name: %s', args.name)
    logging.critical('Destination directory: %s', args.dest)

    if args.mode == 'new':
        logging.critical('Model: %s', args.model)

        model_parameters = {}
        if args.model_dropout:
            model_parameters['dropout'] = args.model_dropout
        if args.model_l2_regularizer:
            model_parameters['l2_regularizer'] = args.model_l2_regularizer

        model = create_model(args.model, model_parameters)
        initial_epoch = 0

    elif args.mode == 'continue':
        logging.critical('Model file: %s', args.model_file)
        logging.critical('Initial epoch: %d', args.initialepoch)
        model = keras.models.load_model(args.model_file)
        initial_epoch = args.initialepoch

    else:
        assert False, 'Unsupported mode {}'.format(args.mode)

    logging.info('Loading data')

    train, validation = data.preload_data(args.datadir, valid_test_size=args.validsize)

    logging.critical('Train set size: %d', len(train))
    logging.critical('Validation set size: %d', len(validation))

    batch_size = args.batchsize
    train_generator = flip_images(generate_data(train, batch_size=batch_size, angle_adj=args.angleadj))
    validation_generator = flip_images(
        generate_data(validation, batch_size=batch_size, angle_adj=args.angleadj))

    model_dir = os.path.join(args.dest, 'model')
    os.makedirs(model_dir, exist_ok=True)
    tf_logs_dir = os.path.join(args.dest, 'tf_logs')
    os.makedirs(tf_logs_dir, exist_ok=True)

    train_model(model,
                train_generator,
                validation_data=validation_generator,
                train_steps_per_epoch=len(train) / batch_size,
                validation_steps_per_epoch=len(validation) / batch_size,
                initial_epoch=initial_epoch,
                epochs=args.epochs,
                tf_logs_dir=tf_logs_dir,
                checkpoints_dir=model_dir,
                name=args.name,
                learning_rate=args.learning_rate)

    model.save(os.path.join(model_dir, '{}-final.h5'.format(args.name)))


if __name__ == '__main__':
    main()
