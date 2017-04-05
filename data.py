import os

import cv2
import keras
import pandas as pd
from sklearn.model_selection import train_test_split


def rmse(y_true, y_pred):
    """
    Metric function: root of mean square error
    """
    return keras.backend.sqrt(keras.metrics.mean_squared_error(y_true, y_pred))


class Indices(object):
    """
    Indices of columns in `driving_log.csv`
    """
    center_image = 0
    left_image = 1
    right_image = 2
    steering = 3


OUTPUT_SHAPE = (160, 320, 3)


def _update_filename_in_row(df, row_idx, col_idx, row, root_dir):
    # Ugh, what's up with indexing in pandas, just want to modify a value in place, how difficult it can be
    new_path = os.path.join(root_dir, 'IMG', os.path.basename(row[col_idx]))
    df.set_value(row_idx, col_idx, new_path)
    return new_path


def preload_data_index(root_dirs, remove_straight_drive_threshold):
    """
    Loads index (`driving_log.csv`) from each root directory, preprocesses, and returns as pandas.DataFrame
    :param remove_straight_drive_threshold: set to remove straight driving longer than certain number of frames
    :param root_dirs: directories to scan
    :return: list of pandas.DataFrame with updated paths
    """
    driving_logs = []
    for root in root_dirs:
        df = pd.read_csv(os.path.join(root, 'driving_log.csv'), header=None)
        df.columns = ['center', 'left', 'right', 'steering', 'throttle', 'brake', 'speed']
        to_remove = []
        for idx, row in df.iterrows():
            new_path = _update_filename_in_row(df, idx, 'center', row, root)
            if not os.path.exists(new_path):
                to_remove.append(idx)
                continue
            _update_filename_in_row(df, idx, 'left', row, root)
            _update_filename_in_row(df, idx, 'right', row, root)
        df.drop(to_remove, inplace=True)
        driving_logs.append(df)

    if not remove_straight_drive_threshold:
        return driving_logs

    for driving_log in driving_logs:
        to_remove = []
        this_iteration = []
        for idx, row in driving_log.iterrows():
            if row['steering'] == 0.0:
                this_iteration.append(idx)
            else:
                if len(this_iteration) > remove_straight_drive_threshold:
                    #print('Found straight drive from {} to {}'.format(this_iteration[0], this_iteration[-1]))
                    to_remove.extend(this_iteration[1:])
                this_iteration = []
        driving_log.drop(to_remove, inplace=True)

    return driving_logs


def preload_data_groupped(root_dirs, valid_test_size, remove_straight_drive_threshold):
    """
    Loads index (`driving_log.csv`) from each root directory, and splits each into train and validation subsets.
    Returns all train sets and validation sets combined together.
    :param root_dirs: list of directories to load the index from
    :param valid_test_size: proportion of validation test in each index
    :return: list with a single tuple of (train_set, validation_set), where each set is a pandas.DataFrame
    """
    data_indices = preload_data_index(root_dirs, remove_straight_drive_threshold)
    data_sets = [train_test_split(data_index, test_size=valid_test_size) for data_index in data_indices]
    train_sets = []
    validation_sets = []
    for train, validation in data_sets:
        train_sets.append(train)
        validation_sets.append(validation)
    return [(pd.concat(train_sets, ignore_index=True), pd.concat(validation_sets, ignore_index=True))]


def preprocess_hist(x):
    yuv = cv2.cvtColor(x, cv2.COLOR_RGB2YUV)
    # clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    clahe = cv2.createCLAHE(clipLimit=3.0)
    cl1 = clahe.apply(yuv[:, :, 0])
    yuv[:, :, 0] = cl1
    # yuv = yuv[trim_top:yuv.shape[0]-trim_bottom, 0:yuv.shape[1], :]
    return yuv


def preprocess_yuv(x):
    yuv = cv2.cvtColor(x, cv2.COLOR_RGB2YUV)
    return yuv
