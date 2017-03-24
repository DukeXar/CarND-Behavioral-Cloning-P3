#!/usr/bin/env python3

import argparse
import os

import pandas as pd


class Indices(object):
    center_image = 0
    left_image = 1
    right_image = 2
    steering = 3


def rename_image(filepath):
    # filepath = /usr/foo/right_2017_03_24_00_05_09_175.jpg

    root_dir = os.path.dirname(filepath)
    original_filename = os.path.basename(filepath)

    filename, ext = os.path.splitext(original_filename)
    items = filename.split('_')

    position = items[0]

    if position not in ['left', 'right', 'center']:
        return filepath

    new_filename = '_'.join(items[1:] + [position]) + ext
    new_filepath = os.path.join(root_dir, new_filename)

    os.rename(filepath, new_filepath)
    #print('Renamed {} to {}'.format(filepath, new_filepath))
    return new_filepath


def normalize_files(datadir):
    log_filename = os.path.join(datadir, 'driving_log.csv')
    driving_log = pd.read_csv(log_filename, header=None)
    os.rename(log_filename, log_filename + '.bak')

    for idx, row in driving_log.iterrows():
        driving_log.set_value(idx, Indices.center_image, rename_image(row[Indices.center_image]))
        driving_log.set_value(idx, Indices.left_image, rename_image(row[Indices.left_image]))
        driving_log.set_value(idx, Indices.right_image, rename_image(row[Indices.right_image]))

    driving_log.to_csv(os.path.join(datadir, 'driving_log.csv'), header=False, index=False)


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('datadir', type=str, help='Directory with data')

    args = parser.parse_args()

    normalize_files(args.datadir)


if __name__ == '__main__':
    main()