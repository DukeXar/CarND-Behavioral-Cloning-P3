import os

import cv2
import keras.backend as K
import keras.metrics as M
import pandas as pd


def rmse(y_true, y_pred):
    return K.sqrt(M.mean_squared_error(y_true, y_pred))


class Indices(object):
    center_image = 0
    left_image = 1
    right_image = 2
    steering = 3


OUTPUT_SHAPE = (160, 320, 3)


def update_filename_in_row(df, row_idx, col_idx, row, root_dir):
    # Ugh, what's up with indexing in pandas, just want to modify a value in place, how difficult it can be
    new_path = os.path.join(root_dir, 'IMG', os.path.basename(row[col_idx]))
    df.set_value(row_idx, col_idx, new_path)
    return new_path


def preload_data_index(root_dirs):
    driving_logs = []
    for root in root_dirs:
        df = pd.read_csv(os.path.join(root, 'driving_log.csv'), header=None)
        df.columns = ['center', 'left', 'right', 'steering', 'throttle', 'brake', 'speed']
        to_remove = []
        for idx, row in df.iterrows():
            new_path = update_filename_in_row(df, idx, 'center', row, root)
            if not os.path.exists(new_path):
                to_remove.append(idx)
                continue
            update_filename_in_row(df, idx, 'left', row, root)
            update_filename_in_row(df, idx, 'right', row, root)
        df.drop(to_remove, inplace=True)
        driving_logs.append(df)
    # ignore_index=True is important, as otherwise it generates duplicates in the result
    # dataset
    combined_log = pd.concat(driving_logs, ignore_index=True)
    return combined_log


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