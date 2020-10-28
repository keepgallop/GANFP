# from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import random
import warnings

import numpy as np
import pandas as pd
from keras.preprocessing.image import ImageDataGenerator

warnings.filterwarnings('ignore')
warnings.simplefilter(action='ignore', category=FutureWarning)


def extract_dataframe_from_dir(dir_path=None, number_per_class=1000, shuffle=True):
    filelist = list(filter(lambda x: x[-1].endswith(('png', 'jpg', 'jpeg')), [[root.split(
        '/')[-1], os.path.join(root, file)] for root, dic, files in os.walk(dir_path) for file in files]))
    if shuffle:
        random.shuffle(filelist)
    y_list = [i[0] for i in filelist]
    x_list = [i[1] for i in filelist]
    classes = list(set(y_list))
    
    count_sum = number_per_class * len(classes)
    assert count_sum <= len(
        x_list), f"Too many images to request. Max {len(x_list)}, got {count_sum}"
    
    selected_x_list, selected_y_list = [], []
    for c in classes:
        count = 0
        for i, label in enumerate(y_list):
            if label == c:
                selected_x_list.append(x_list[i])
                selected_y_list.append(y_list[i])
                count += 1
            if count == number_per_class:
                break
                  
    img_dataframe = pd.DataFrame({'filename': selected_x_list,
                                  'class': selected_y_list})
    return img_dataframe


def data_gen(path=None,
             preprocessing_function=None,
             target_size=(128, 128),
             batch_size=64,
             color_mode='rgb',
             shuffle=True,
             source="directory",
             dataframe=None,
             ):
    image_loader_gen = ImageDataGenerator(
        preprocessing_function=preprocessing_function)
    if source == "directory":
        if color_mode in ['rgb', 'ycc']:
            image_gen = image_loader_gen.flow_from_directory(
                path, target_size=(128, 128), batch_size=batch_size, color_mode='rgb', shuffle=shuffle)
        else:
            image_gen = image_loader_gen.flow_from_directory(path, target_size=(
                128, 128), batch_size=batch_size, color_mode='grayscale', shuffle=shuffle)

    elif source == "dataframe":
        if color_mode in ['rgb', 'ycc']:
            image_gen = image_loader_gen.flow_from_dataframe(
                dataframe, target_size=(128, 128), batch_size=batch_size, color_mode='rgb', shuffle=shuffle)
        else:
            image_gen = image_loader_gen.flow_from_dataframe(dataframe, target_size=(
                128, 128), batch_size=batch_size, color_mode='gray-scale', shuffle=shuffle)

    return image_gen


def attack_gen(path):
    pass


def stats_from_generator(gen, steps=1, stats_axis=(0, 1, 2)):
    means, stds, maxs, mins = [], [], [], []

    for i in range(steps):
        print(f"compute batch {i}/{steps}")
        batch = next(gen)[0]
        means.append(np.mean(batch, stats_axis))
        stds.append(np.std(batch, stats_axis))
        maxs.append(np.max(np.abs(batch), stats_axis))
        mins.append(np.min(np.abs(batch), stats_axis))
    means, stds, maxs, mins = list(map(np.asarray, [means, stds, maxs, mins]))
    return np.mean(means, axis=0), np.mean(stds, axis=0), np.max(maxs, axis=0), np.min(mins, axis=0)
