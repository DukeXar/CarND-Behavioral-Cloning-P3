import argparse
import os

import numpy as np
import pandas as pd
import skimage.io
import sklearn

from keras.callbacks import ModelCheckpoint
from keras.callbacks import TensorBoard
from keras.layers import Conv2D, MaxPooling2D, Cropping2D
from keras.layers import Flatten, Dense, Lambda
from keras.models import Sequential
from sklearn.model_selection import train_test_split
import sklearn.utils


def create_model_lenet():
    model = Sequential()
    model.add(Cropping2D(cropping=((50,20), (0,0)), input_shape=(160,320,3)))
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
    model.add(Cropping2D(cropping=((50,20), (0,0)), input_shape=(160,320,3)))
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


def create_model(name):
    models = {'lenet': create_model_lenet, 'nvidia': create_model_nvidia}
    return models[name]()


def train_model(model,
                train_generator,
                validation_generator,
                train_steps_per_epoch,
                validation_steps_per_epoch,
                initial_epoch,
                epochs,
                tf_logs_dir,
                checkpoints_dir,
                name):
    tensorboard_callback = TensorBoard(log_dir=tf_logs_dir, histogram_freq=0,
                                       write_graph=True, write_images=False)

    filepath = os.path.join(checkpoints_dir,
                            'model-' + name + '-improvement-{epoch:02d}-{val_loss:.3f}.h5')
    checkpoint_callback = ModelCheckpoint(filepath, monitor='val_loss',
                                          verbose=0, save_weights_only=False,
                                          save_best_only=True, mode='max')

    model.compile(loss='mse', optimizer='adam')
    model.fit_generator(train_generator,
                        steps_per_epoch=train_steps_per_epoch,
                        epochs=epochs,
                        initial_epoch=initial_epoch,
                        validation_data=validation_generator,
                        validation_steps=validation_steps_per_epoch,
                        callbacks=[tensorboard_callback, checkpoint_callback])


def generate_data(samples, root_dir, batch_size=32):
    """
    Generator to lazily load the data.
    :param samples: pandas.DataFrame
    :param batch_size: batch size
    :return: generator
    """

    def get_filename(filename):
        return os.path.join(root_dir, 'IMG', os.path.basename(filename))

    output_shape = (160, 320, 3)
    angle_adj = 0.1

    steering_idx = 4
    center_image_idx = 1
    left_image_idx = 2
    right_image_idx = 3

    images = np.empty((batch_size,) + output_shape, np.float32)
    angles = np.empty((batch_size,), np.float32)

    def gen_data(batch_samples, filename_idx, angle_adj):
        for idx, row in enumerate(batch_samples.itertuples()):
            images[idx] = skimage.io.imread(get_filename(row[filename_idx]))
            angles[idx] = row[steering_idx] + angle_adj

        yield sklearn.utils.shuffle(images, angles)

    while True:
        sklearn.utils.shuffle(samples)

        for offset in range(0, len(samples), batch_size):
            batch_samples = samples[offset:offset+batch_size]
            yield from gen_data(batch_samples, center_image_idx, 0)
            yield from gen_data(batch_samples, left_image_idx, angle_adj)
            yield from gen_data(batch_samples, right_image_idx, -angle_adj)


def flip_images(batch_generator):
    for batch_images, batch_angles in batch_generator:
        images = batch_images.copy()
        angles = batch_angles.copy()
        indices = np.random.randint(0, len(batch_images), int(len(batch_images)/2))
        for idx in indices:
            images[idx] = np.fliplr(images[idx])
            angles[idx] = -angles[idx]
        yield images, angles


def preload_data(root_dir):
    driving_log = pd.read_csv(os.path.join(root_dir, 'driving_log.csv'))
    train, validation = train_test_split(driving_log, test_size=0.2)
    return train, validation


def main():
    root_dir = './data2-3'
    batch_size = 32
    model_name = 'nvidia'

    train, validation = preload_data(root_dir)
    train_generator = flip_images(generate_data(train, root_dir, batch_size=batch_size))
    validation_generator = flip_images(generate_data(validation, root_dir, batch_size=batch_size))

    model = create_model(model_name)

    train_model(model,
                train_generator,
                validation_generator,
                train_steps_per_epoch=len(train)/batch_size,
                validation_steps_per_epoch=len(validation)/batch_size,
                initial_epoch=0,
                epochs=50,
                tf_logs_dir='./logs',
                checkpoints_dir='./',
                name=model_name)

    model.save('model-{}.h5'.format(model_name))


if __name__ == '__main__':
    main()
