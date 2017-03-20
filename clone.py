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


def create_model():
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


def train_model(model, X_train, y_train, initial_epoch, epochs, tf_logs_dir,
                checkpoints_dir):
    tensorboard_callback = TensorBoard(log_dir=tf_logs_dir, histogram_freq=0,
                                       write_graph=True, write_images=False)

    filepath = os.path.join(checkpoints_dir,
                            'model-improvement-{epoch:02d}-{val_loss:.2f}.h5')
    checkpoint_callback = ModelCheckpoint(filepath, monitor='val_loss',
                                          verbose=1, save_weights_only=False,
                                          save_best_only=True, mode='max')

    model.compile(loss='mse', optimizer='adam')
    model.fit(X_train, y_train, validation_split=0.2, shuffle=True,
              nb_epoch=epochs, initial_epoch=initial_epoch,
              callbacks=[tensorboard_callback, checkpoint_callback])


def load_data(root_dir):
    driving_log = pd.read_csv(os.path.join(root_dir, 'driving_log.csv'))

    images = []
    for source_path in driving_log['center']:
        filename = os.path.basename(source_path)
        current_path = os.path.join(root_dir, 'IMG', filename)
        image = skimage.io.imread(current_path)
        images.append(image)

    X_train = np.array(images)
    y_train = np.array(driving_log['steering'])
    return X_train, y_train


def main():
    X_train, y_train = load_data('./data')
    model = create_model()
    train_model(model, X_train, y_train, initial_epoch=0, epochs=10,
                tf_logs_dir='./logs', checkpoints_dir='./')
    model.save('model.h5')


if __name__ == '__main__':
    main()
