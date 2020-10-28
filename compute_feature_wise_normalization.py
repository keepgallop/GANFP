# Compute configures for feature wise normalization using training dataset

import argparse
import functools
import os
import warnings

import joblib
# import tensorflow as tf
import keras
import pandas as pd

import image_generator
import nets
import transformation as T
import utils
from itertools import product
def compute_feature_wise_normalization(dataset_name, 
                                       traindatasetpath, 
                                       color_mode, 
                                       feature_space, 
                                       rescale, 
                                       logscale, 
                                       batch_size = 200, 
                                       steps = 200):
    # parameters for data generator

    # preprocessing pipeline
    placeholder_func = functools.partial(lambda x: x)

    # 1. color mode
    if color_mode in ['gray', 'rgb']:
        COLOR_TRANS = placeholder_func
    else:
        COLOR_TRANS = functools.partial(T.rgb_to_ycc)

    # 2. feature space
    if feature_space == 'raw':
        SPACE_TRANS = placeholder_func
        if not rescale:
            SCALE = placeholder_func
        else:
            SCALE = functools.partial(T.rescale, rescale_factor=255.)
    else:
        SPACE_TRANS = functools.partial(T.dct2)
        if not logscale:
            SCALE = placeholder_func
        else:
            SCALE = functools.partial(T.log_scale, epsilon=1e-12)

    # 3. normalize
    if not os.path.exists('./normalize_conf.conf'):
        conf_dic = dict({'args': [], 'normalize_factor': []})
        joblib.dump(conf_dic, './normalize_conf.conf')
    
    current_args = f"{dataset_name}-{color_mode}-{feature_space}-r{rescale}-l{logscale}-"
    conf_dic = joblib.load('./normalize_conf.conf')

    if current_args not in conf_dic['args']:
        print(current_args)
        tmp_funcs_pipeline = utils.compose_functions(
            COLOR_TRANS, SPACE_TRANS, SCALE)
        tmp_gen = image_generator.data_gen(
            traindatasetpath, tmp_funcs_pipeline, batch_size=batch_size, color_mode=color_mode)
        n_mean, n_std, n_max, n_min = image_generator.stats_from_generator(
            tmp_gen, steps)

        conf_dic['args'].append(current_args)
        conf_dic['normalize_factor'].append(
            [n_mean, n_std, n_max, n_min])
        joblib.dump(conf_dic, './normalize_conf.conf')

def main():
    utils.clean_fk_file('.', '.DS_Store')
    utils.clean_fk_file('.', '.ipynb_checkpoints')
    utils.clean_useless_image('.')
    for dataset_name, traindatasetpath in zip(['CelebA', 'LSUN'], ["./dataset/CelebA/Train", "./dataset/LSUN/Train"]):
        for color_mode, feature_space in product(['gray', 'rgb', 'ycc'], ['raw', 'dct']):
            if feature_space == 'raw':
                rescale = True
                logscale = False
            else:
                rescale = False
                logscale = True

            compute_feature_wise_normalization(dataset_name, 
                                               traindatasetpath, 
                                               color_mode, 
                                               feature_space, 
                                               rescale, 
                                               logscale)
            

if "__main__" == __name__:
    main()