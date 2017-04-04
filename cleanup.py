#!/usr/bin/env python3

import os
import argparse


def process_subdirectory(model_directory, keep_n, dry_run):
    all_models = sorted([fname for fname in os.listdir(model_directory) if fname.endswith('.h5')])
    intermediate_models = [fname for fname in all_models if not fname.endswith('-final.h5')]

    print('Found intermediate models: {}'.format(intermediate_models))

    for model in intermediate_models[:-keep_n]:
        full_path = os.path.join(model_directory, model)
        if dry_run:
            print('Not removing (dry-run) {}'.format(full_path))
        else:
            print('Removing {}'.format(full_path))
            os.remove(full_path)


def process_root(root, keep_n, dry_run):
    print('Processing {}'.format(root))

    model_directory = os.path.join(root, 'model')
    if not os.path.isdir(model_directory):
        print('Unsupported root {}: {} is not a directory'.format(root, model_directory))
        return

    all_folds = [dirname for dirname in os.listdir(model_directory) if dirname.startswith('fold_') and os.path.isdir(dirname)]

    if all_folds:
        for fold in all_folds:
            process_subdirectory(os.path.join(model_directory, fold), keep_n, dry_run)
    else:
        process_subdirectory(model_directory, keep_n, dry_run)


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('root', type=str, nargs='+', help='Directory where ./model is located')
    parser.add_argument('--dry-run', action='store_true', default=False, help='Report, but do not remove')
    parser.add_argument('--keep-n', type=int, default=1, help='How many intermediate models to keep')

    args = parser.parse_args()

    print('Processing {} keeping {} intermediate models'.format(args.root, args.keep_n))

    for root in args.root:
        process_root(root, args.keep_n, args.dry_run)

    print('Done')

if __name__ == '__main__':
    main()