import os

import pandas as pd
from sklearn.model_selection import train_test_split


class Indices(object):
    center_image = 0
    left_image = 1
    right_image = 2
    steering = 3


OUTPUT_SHAPE = (160, 320, 3)


def update_filename_in_row(df, row_idx, col_idx, row, root_dir):
    # Ugh, what's up with indexing in pandas, just want to modify a value in place, how difficult it can be
    df.set_value(row_idx, col_idx, os.path.join(root_dir, 'IMG', os.path.basename(row[col_idx])))


def preload_data(root_dirs, valid_test_size=0.2):
    driving_logs = []
    for root in root_dirs:
        df = pd.read_csv(os.path.join(root, 'driving_log.csv'), header=None)
        df.columns = ['center', 'left', 'right', 'steering', 'throttle', 'brake', 'speed']
        for idx, row in df.iterrows():
            update_filename_in_row(df, idx, 'center', row, root)
            update_filename_in_row(df, idx, 'left', row, root)
            update_filename_in_row(df, idx, 'right', row, root)
        driving_logs.append(df)
    combined_log = pd.concat(driving_logs)

    train, validation = train_test_split(combined_log, test_size=valid_test_size)
    return train, validation
