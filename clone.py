#!/usr/bin/env python3

import argparse
import os
import random
from datetime import datetime

import numpy as np
import pandas as pd
import skimage.io
import sklearn
import sklearn.utils
from keras.callbacks import ModelCheckpoint
from keras.callbacks import TensorBoard
from keras.layers import Conv2D, MaxPooling2D, Cropping2D
from keras.layers import Flatten, Dense, Lambda
from keras.models import Sequential
from sklearn.model_selection import train_test_split


def create_model_lenet():
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


def create_model_nvidia():
    model = Sequential()
    model.add(Cropping2D(cropping=((50, 20), (0, 0)), input_shape=(160, 320, 3)))
    model.add(Lambda(lambda x: x / 255.0 - 0.5))
    model.add(Conv2D(24, (5, 5), strides=(2, 2), activation='relu'))
    model.add(Conv2D(36, (5, 5), strides=(2, 2), activation='relu'))
    model.add(Conv2D(48, (5, 5), strides=(2, 2), activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(Flatten())
    model.add(Dense(100))
    model.add(Dense(50))
    model.add(Dense(10))
    model.add(Dense(1))
    return model


MODELS = {
    'lenet': create_model_lenet,
    'nvidia': create_model_nvidia
}

DEFAULT_MODEL = 'nvidia'


def create_model(name):
    return MODELS[name]()


def train_model(model,
                train_generator,
                validation_data,
                train_steps_per_epoch,
                validation_steps_per_epoch,
                initial_epoch,
                epochs,
                tf_logs_dir,
                checkpoints_dir,
                name):
    tensorboard_callback = TensorBoard(log_dir=tf_logs_dir, histogram_freq=1,
                                       write_graph=True, write_images=True)

    filepath = os.path.join(checkpoints_dir,
                            'model-' + name + '-improved-{epoch:02d}-{val_loss:.3f}.h5')

    checkpoint_callback = ModelCheckpoint(filepath, monitor='val_loss',
                                          verbose=0, save_weights_only=False,
                                          save_best_only=True)

    model.compile(loss='mse', optimizer='adam')
    model.fit_generator(train_generator,
                        steps_per_epoch=train_steps_per_epoch,
                        epochs=epochs,
                        initial_epoch=initial_epoch,
                        validation_data=validation_data,
                        validation_steps=validation_steps_per_epoch,
                        callbacks=[tensorboard_callback, checkpoint_callback])


class Indices(object):
    center_image = 0
    left_image = 1
    right_image = 2
    steering = 3


def generate_data(samples, batch_size=32, angle_adj=0.1, augment=True):
    output_shape = (160, 320, 3)

    images = np.empty((batch_size,) + output_shape, np.float32)
    angles = np.empty((batch_size,), np.float32)

    def gen_data(a_batch_samples, filename_idx, an_angle_adj):
        for idx, row in enumerate(a_batch_samples.itertuples()):
            images[idx] = skimage.io.imread(row[filename_idx])
            angles[idx] = row[Indices.steering+1] + an_angle_adj

        yield sklearn.utils.shuffle(images, angles)

    while True:
        sklearn.utils.shuffle(samples)

        for offset in range(0, len(samples), batch_size):
            batch_samples = samples[offset:offset + batch_size]
            yield from gen_data(batch_samples, Indices.center_image+1, 0)

            if augment:
                yield from gen_data(batch_samples, Indices.left_image+1, angle_adj)
                yield from gen_data(batch_samples, Indices.right_image+1, -angle_adj)


def flip_images(batch_generator, flip_prob=0.5):
    for batch_images, batch_angles in batch_generator:
        images = batch_images.copy()
        angles = batch_angles.copy()
        indices = random.sample(range(0, len(batch_images)), int(len(batch_images) * flip_prob))
        for idx in indices:
            images[idx] = np.fliplr(images[idx])
            angles[idx] = -angles[idx]
        yield images, angles


def update_filename_in_row(df, row_idx, col_idx, row, root_dir):
    # Ugh, what's up with indexing in pandas, just want to modify a value in place, how difficult it can be
    df.set_value(row_idx, col_idx, os.path.join(root_dir, 'IMG', os.path.basename(row[col_idx])))


def preload_data(root_dirs, valid_test_size=0.2):
    driving_logs = []
    for root in root_dirs:
        df = pd.read_csv(os.path.join(root, 'driving_log.csv'), header=None)
        for idx, row in df.iterrows():
            update_filename_in_row(df, idx, Indices.center_image, row, root)
            update_filename_in_row(df, idx, Indices.left_image, row, root)
            update_filename_in_row(df, idx, Indices.right_image, row, root)
        driving_logs.append(df)
    combined_log = pd.concat(driving_logs)

    train, validation = train_test_split(combined_log, test_size=valid_test_size)
    return train, validation


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('datadir', type=str, nargs='+', help='Directories with data')
    parser.add_argument('--model', type=str, choices=MODELS.keys(), default=DEFAULT_MODEL, help='Model type')
    parser.add_argument('--batchsize', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--initialepoch', type=int, default=0)
    parser.add_argument('--name', type=str, default='noname', help='Model name')
    parser.add_argument('--dest', type=str, default=os.path.join('out', datetime.now().isoformat()),
                        help='Destination directory')
    parser.add_argument('--angleadj', type=float, default=0.1, help='Angle adjustment for left-right images')
    parser.add_argument('--validsize', type=float, default=0.2, help='Validation test size')

    args = parser.parse_args()

    batch_size = args.batchsize

    train, validation = preload_data(args.datadir, valid_test_size=args.validsize)
    train_generator = flip_images(generate_data(train, batch_size=batch_size, angle_adj=args.angleadj))
    validation_generator = flip_images(
        generate_data(validation, batch_size=batch_size, angle_adj=args.angleadj))

    model = create_model(args.model)

    model_dir = os.path.join(args.dest, 'model')
    os.makedirs(model_dir, exist_ok=True)
    tf_logs_dir = os.path.join(args.dest, 'tf_logs')
    os.makedirs(tf_logs_dir, exist_ok=True)

    train_model(model,
                train_generator,
                validation_data=validation_generator,
                train_steps_per_epoch=len(train) / batch_size,
                validation_steps_per_epoch=len(validation) / batch_size,
                initial_epoch=args.initialepoch,
                epochs=args.epochs,
                tf_logs_dir=tf_logs_dir,
                checkpoints_dir=model_dir,
                name=args.name)

    model.save(os.path.join(model_dir, '{}-final.h5'.format(args.name)))


if __name__ == '__main__':
    main()
