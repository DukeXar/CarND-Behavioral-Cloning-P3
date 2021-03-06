#!/usr/bin/env python3

import argparse
import functools
import logging
import os
import socket
import sys
from datetime import datetime

import keras.models
import keras.regularizers
import keras.utils
import numpy as np
import pandas as pd
import skimage.color
import skimage.io
import skimage.transform
import sklearn
import sklearn.utils
from keras.applications import VGG16
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.callbacks import TensorBoard
from keras.layers import Conv2D, Cropping2D
from keras.layers import Flatten, Dense, Lambda, Dropout, Input
from keras.models import Sequential, Model
from sklearn.model_selection import KFold

import data


def create_model_nvidia_4(parameters):
    dropout = parameters.get('dropout', 0)
    l2_regularizer_lambda = parameters.get('l2_regularizer', 0)

    if l2_regularizer_lambda:
        l2_regularizer = keras.regularizers.l2(l2_regularizer_lambda)
    else:
        l2_regularizer = None

    model = Sequential()
    model.add(Cropping2D(cropping=((74, 20), (35, 35)), input_shape=(160, 320, 3)))
    model.add(Lambda(lambda x: x / 255.0 - 0.5))
    model.add(Conv2D(24, (5, 5), strides=(2, 2), activation='relu'))
    model.add(Conv2D(36, (5, 5), strides=(2, 2), activation='relu'))
    model.add(Conv2D(48, (5, 5), strides=(2, 2), activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(Flatten())

    if dropout:
        model.add(Dropout(rate=0.05))
    model.add(Dense(100, activation='relu', kernel_regularizer=l2_regularizer, kernel_initializer='he_uniform'))
    if dropout:
        model.add(Dropout(rate=0.2))
    model.add(Dense(50, activation='relu', kernel_regularizer=l2_regularizer, kernel_initializer='he_uniform'))
    if dropout:
        model.add(Dropout(rate=dropout))
    model.add(Dense(10, activation='relu', kernel_regularizer=l2_regularizer, kernel_initializer='he_uniform'))
    if dropout:
        model.add(Dropout(rate=dropout))
    model.add(Dense(1, kernel_regularizer=l2_regularizer, kernel_initializer='he_uniform'))
    return model


def create_model_commaai(parameters):
    dropout = parameters.get('dropout', 0)
    l2_regularizer_lambda = parameters.get('l2_regularizer', 0)

    if l2_regularizer_lambda:
        l2_regularizer = keras.regularizers.l2(l2_regularizer_lambda)
    else:
        l2_regularizer = None

    model = Sequential()
    model.add(Cropping2D(cropping=((74, 20), (35, 35)), input_shape=(160, 320, 3)))
    model.add(Lambda(lambda x: x / 255.0 - 0.5))

    model.add(Conv2D(16, (8, 8), strides=(4, 4), padding='same', activation='elu'))
    model.add(Conv2D(32, (5, 5), strides=(2, 2), padding='same', activation='elu'))
    model.add(Conv2D(64, (5, 5), strides=(2, 2), padding='same', activation='elu'))
    model.add(Flatten())

    if dropout:
        model.add(Dropout(rate=0.2))  # 0.2
    model.add(Dense(512, activation='elu'))
    if dropout:
        model.add(Dropout(rate=dropout))  # 0.5
    model.add(Dense(1))
    return model


def create_model_vgg16(parameters):
    dropout = parameters.get('dropout', 0)
    l2_regularizer_lambda = parameters.get('l2_regularizer', 0)

    if l2_regularizer_lambda:
        l2_regularizer = keras.regularizers.l2(l2_regularizer_lambda)
    else:
        l2_regularizer = None

    input = Input(shape=(160, 320, 3))
    x = Cropping2D(cropping=((59, 35), (60, 60)))(input)
    x = Lambda(lambda x: x / 255.0 - 0.5)(x)

    vgg16_conv = VGG16(include_top=False, weights='imagenet', input_tensor=x)
    for layer in vgg16_conv.layers:
        layer.trainable = False

    x = Flatten(name='flatten')(vgg16_conv.output)
    if dropout:
        x = Dropout(rate=dropout)(x)
    x = Dense(4096, activation='relu', name='fc1', kernel_regularizer=l2_regularizer)(x)
    if dropout:
        x = Dropout(rate=dropout)(x)
    x = Dense(4096, activation='relu', name='fc2', kernel_regularizer=l2_regularizer)(x)
    if dropout:
        x = Dropout(rate=dropout)(x)
    x = Dense(1, name='predictions', kernel_regularizer=l2_regularizer)(x)

    model = Model(inputs=input, outputs=x)
    return model


MODELS = {
    'vgg16': create_model_vgg16,
    'nvidia4': create_model_nvidia_4,
    'commaai': create_model_commaai
}

DEFAULT_MODEL = 'nvidia'


def create_model(name, parameters):
    logging.info('Creating model %s, parameters=%s', name, parameters)
    return MODELS[name](parameters)


class TrainLogCallback(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if logs is None:
            logs = {}

        items = ", ".join("{}={}".format(k, v) for k, v in logs.items())
        logging.info('Epoch {}/{} end: {}'.format(epoch + 1, self.params['epochs'], items))

    def on_train_end(self, logs=None):
        logging.info('Train end: {}'.format(logs))


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
                learning_rate,
                early_stopping_patience,
                workers,
                verbose):
    optimizer = keras.optimizers.Adam(lr=learning_rate)
    loss_function = 'mse'

    logging.info(('Training model {}: epochs={}, initial_epoch={}, train_steps_per_epoch={}, '
                  'validation_steps_per_epoch={}, loss={}, optimizer={}, early_stopping_patience={}, '
                  'learning_rate={}').format(name, epochs, initial_epoch, train_steps_per_epoch,
                                             validation_steps_per_epoch, loss_function, optimizer,
                                             early_stopping_patience, learning_rate))

    callbacks = []
    tensorboard_callback = TensorBoard(log_dir=tf_logs_dir, histogram_freq=1,
                                       write_graph=True, write_images=True)
    callbacks.append(tensorboard_callback)

    filepath = os.path.join(checkpoints_dir,
                            'model-' + name + '-improved-{epoch:02d}-{val_loss:.3f}.h5')

    checkpoint_callback = ModelCheckpoint(filepath, monitor='val_loss',
                                          verbose=1, save_weights_only=False,
                                          save_best_only=True)
    callbacks.append(checkpoint_callback)

    if early_stopping_patience is not None:
        early_stopping_callback = EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=early_stopping_patience,
                                                verbose=1)
        callbacks.append(early_stopping_callback)

    train_log_callback = TrainLogCallback()
    callbacks.append(train_log_callback)

    model.compile(loss=loss_function, optimizer=optimizer, metrics=[data.rmse])
    model.fit_generator(train_generator,
                        steps_per_epoch=train_steps_per_epoch,
                        epochs=epochs,
                        initial_epoch=initial_epoch,
                        validation_data=validation_data,
                        validation_steps=validation_steps_per_epoch,
                        callbacks=callbacks,
                        verbose=verbose,
                        workers=workers)


def _process_image_inline(row, images, idx, filename_idx, enable_preprocess_hist, enable_preprocess_yuv,
                          enable_preprocess_hsv):
    """Helper to load image from row, preprocess and write it back into the images[idx]"""
    image = skimage.io.imread(row[filename_idx])
    if enable_preprocess_hist:
        images[idx] = data.preprocess_hist(image)
    elif enable_preprocess_yuv:
        images[idx] = data.preprocess_yuv(image)
    elif enable_preprocess_hsv:
        images[idx] = skimage.color.rgb2hsv(image)
    else:
        images[idx] = image


def generate_data(samples, batch_size, angle_adj, random_state=None,
                  enable_preprocess_hist=False, enable_preprocess_yuv=False, enable_preprocess_hsv=False):
    """Training and validation data generator. It generates images on the fly from the `samples` dataset, which contains
    only filenames. It outputs batches of `batch_size`.
    :param samples - input dataset with filesnames
    :param batch_size - output batch size
    :param angle_adj - when non zero, will use left and right camera images to generate data point with angle adjusted
                       to this value
    :param random_state - can be useful to generate reproducible data sets
    :param enable_preprocess_hist - enable image preprocessing by applying adaptive histogram adjustment
    :param enable_preprocess_yuv - preprocess image by converting into YUV colorspace
    :param enable_preprocess_hsv - preprocess image by converting into HSV colorspace
    """
    def gen_data(a_batch_samples, filename_idx, an_angle_adj):
        # This loads and preprocesses one batch
        images = np.empty((len(a_batch_samples),) + data.OUTPUT_SHAPE, np.float32)
        angles = np.empty((len(a_batch_samples),), np.float32)

        for idx, row in enumerate(a_batch_samples.itertuples()):
            _process_image_inline(row, images, idx, filename_idx, enable_preprocess_hist, enable_preprocess_yuv,
                                  enable_preprocess_hsv)

        for idx, row in enumerate(a_batch_samples.itertuples()):
            angles[idx] = row[data.Indices.steering + 1] + an_angle_adj

        return sklearn.utils.shuffle(images, angles)

    if random_state is not None:
        np.random.seed(random_state)

    current_samples = samples
    while True:
        current_samples = sklearn.utils.shuffle(current_samples)

        for offset in range(0, len(current_samples), batch_size):
            batch_samples = current_samples[offset:min(len(current_samples), offset + batch_size)]

            if angle_adj is not None:
                mode = np.random.random_integers(1, 3)
            else:
                mode = 1

            if mode == 1:
                batch = gen_data(batch_samples, data.Indices.center_image + 1, 0)
            elif mode == 2:
                batch = gen_data(batch_samples, data.Indices.left_image + 1, angle_adj)
            else:
                batch = gen_data(batch_samples, data.Indices.right_image + 1, -angle_adj)

            do_flip = np.random.random_integers(0, 1)

            if do_flip:
                images, angles = batch[0].copy(), batch[1].copy()
                angles = -angles
                for i in range(len(images)):
                    images[i] = np.fliplr(images[i])

                batch = images, angles

            yield batch


def log_model_summary(model):
    """Helper hack to dump model summary into the log file"""
    keras.utils.layer_utils.print = logging.critical
    keras.utils.layer_utils.print_summary(model)
    keras.utils.layer_utils.print = print


def parse_arguments():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--verbose', type=int, default=2, help='Keras verbosity')
    parser.add_argument('--workers', type=int, default=1, help='Number of Keras workers')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs to learn')
    parser.add_argument('--learning-rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--early-stopping-patience', type=int, default=None, help='Early stopping patience in epochs')
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
    data_group.add_argument('--kfolds', type=int, default=None, help='Use K-Fold')
    data_group.add_argument('--preprocess-hist', action='store_true',
                            help='Enable yuv conversion and histogram equalization')
    data_group.add_argument('--preprocess-yuv', action='store_true', help='Enable yuv conversion')
    data_group.add_argument('--preprocess-hsv', action='store_true', help='Enable hsv conversion')
    data_group.add_argument('--remove-straight-drive-threshold', type=float, default=0.0,
                            help='Proportion of straight driving frames to remove')

    new_model_parser.add_argument('datadir', type=str, nargs='+', help='Directories with data')
    cont_model_parser.add_argument('datadir', type=str, nargs='+', help='Directories with data')

    args = parser.parse_args()
    return args


def main():
    args = parse_arguments()

    logging_filename = os.path.join(args.dest, 'clone.log')
    os.makedirs(os.path.dirname(logging_filename), exist_ok=True)

    from tensorflow.python.platform import tf_logging
    tf_logging._logger.removeHandler(tf_logging._handler)

    logging.basicConfig(filename=logging_filename,
                        level=logging.INFO,
                        format='%(asctime)s %(levelname)s %(message)s')

    logging.critical('Starting cloning')
    logging.critical('Started as: %s', ' '.join(sys.argv))
    logging.critical('Data directories: %s', args.datadir)
    logging.critical('Batch size: %d', args.batchsize)
    logging.critical('Learning rate: %s', args.learning_rate)
    logging.critical('Epochs: %d', args.epochs)
    logging.critical('Early stopping patience: %s', args.early_stopping_patience)
    logging.critical('Angle adjustment: %s', args.angleadj)
    logging.critical('Histogram equalization and YUV: %s', args.preprocess_hist)
    logging.critical('YUV preprocessing: %s', args.preprocess_yuv)
    logging.critical('Remove straight drive threshold: %s', args.remove_straight_drive_threshold)
    logging.critical('Validation set size: %s', args.validsize)
    logging.critical('Model name: %s', args.name)
    logging.critical('Destination directory: %s', args.dest)
    logging.critical('Number of workers: %d', args.workers)

    logging.info('Loading data')

    if args.kfolds is not None:
        logging.critical('K-Folds: %s', args.kfolds)

        # ignore_index=True is important, as otherwise it generates duplicates in the result
        # dataset
        data_index = pd.concat(data.preload_data_index(args.datadir, args.remove_straight_drive_threshold),
                               ignore_index=True)

        # KFold returns an array with indices, use those indices to retrieve actual items
        # It should be cheap, as our items are just image paths and some float values.
        def select_data(folds_gen):
            for fold in folds_gen:
                train_index, validation_index = fold
                train = data_index.loc[train_index, :]
                validation = data_index.loc[validation_index, :]
                yield train, validation

        folds = select_data(KFold(n_splits=args.kfolds, shuffle=False).split(data_index))
    else:
        logging.critical('Validation size: %s', args.validsize)
        folds = data.preload_data_groupped(args.datadir, args.validsize, args.remove_straight_drive_threshold)

    if args.mode == 'new':
        logging.critical('Model: %s', args.model)

        model_parameters = {}
        if args.model_dropout:
            model_parameters['dropout'] = args.model_dropout
        if args.model_l2_regularizer:
            model_parameters['l2_regularizer'] = args.model_l2_regularizer

        model = create_model(args.model, model_parameters)
        recreate_model = functools.partial(create_model, name=args.model, parameters=model_parameters)
        log_model_summary(model)
        initial_epoch = 0

    elif args.mode == 'continue':
        logging.critical('Model file: %s', args.model_file)
        logging.critical('Initial epoch: %d', args.initialepoch)
        model = keras.models.load_model(args.model_file, custom_objects={'rmse': data.rmse})
        recreate_model = functools.partial(keras.models.load_model, args.model_file)
        initial_epoch = args.initialepoch
        log_model_summary(model)

    else:
        assert False, 'Unsupported mode {}'.format(args.mode)

    for fold_idx, fold in enumerate(folds):
        if fold_idx > 0:
            model = recreate_model()

        train, validation = fold
        k_folds = 1 if args.kfolds is None else args.kfolds
        logging.critical('Running fold %d/%d', fold_idx + 1, k_folds)
        logging.critical('Train set size: %d', len(train))
        logging.critical('Validation set size: %d', len(validation))

        model_dir = os.path.join(args.dest, 'model', 'fold_{:02d}'.format(fold_idx + 1))
        os.makedirs(model_dir, exist_ok=True)
        tf_logs_dir = os.path.join(args.dest, 'tf_logs', 'fold_{:02d}'.format(fold_idx + 1))
        os.makedirs(tf_logs_dir, exist_ok=True)

        batch_size = args.batchsize
        train_generator = generate_data(train, batch_size=batch_size, angle_adj=args.angleadj,
                                        enable_preprocess_hist=args.preprocess_hist,
                                        enable_preprocess_yuv=args.preprocess_yuv)
        validation_generator = generate_data(validation, batch_size=batch_size, angle_adj=args.angleadj,
                                             enable_preprocess_hist=args.preprocess_hist,
                                             enable_preprocess_yuv=args.preprocess_yuv)

        # Depending on the augmentation, number of batches is different to feed all possible augmentations into the
        # model.
        if args.angleadj is not None:
            # 3 angles * 2 flips
            scale = 6 / batch_size
        else:
            # 2 flips
            scale = 2 / batch_size

        train_steps_per_epoch = int(len(train) * scale)
        validation_steps_per_epoch = int(len(validation) * scale)

        train_model(model,
                    train_generator,
                    validation_data=validation_generator,
                    train_steps_per_epoch=train_steps_per_epoch,
                    validation_steps_per_epoch=validation_steps_per_epoch,
                    initial_epoch=initial_epoch,
                    epochs=args.epochs,
                    tf_logs_dir=tf_logs_dir,
                    checkpoints_dir=model_dir,
                    name=args.name,
                    learning_rate=args.learning_rate,
                    early_stopping_patience=args.early_stopping_patience,
                    workers=args.workers,
                    verbose=args.verbose)

        # Just in case, so we can continue.
        model.save(os.path.join(model_dir, '{}-final.h5'.format(args.name)))


if __name__ == '__main__':
    main()
