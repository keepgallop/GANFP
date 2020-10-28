import numpy as np
import shutil 
import os
import functools
from PIL import Image
import keras.backend as K
def load_img_arr_128(im_path, color_mode = 'rgb'):
    im = Image.open(im_path)
    if color_mode == 'gray':
        im = im.convert('L')
        shape = (128,128)
    else:
        shape = (128,128,3)
    
    im = np.array(im)
    if im.shape != shape:
        im.resize(shape)
    return im
 
def round_thre(arr):
    return np.round(arr, decimals=1)
     
def cutoff_thre(arr, cutoff):
    assert isinstance(cutoff, (float, tuple))
    if isinstance(cutoff, float):
        assert 0 < cutoff < 1
        acreq = arr.copy()
        acreq[acreq > cutoff] = 1
        acreq[acreq <= cutoff] = 0
        return acreq
    else:
        assert len(cutoff) == 2
        assert 0 < cutoff[0] < cutoff[1] < 1
        conditions = [arr <= cutoff[0], (cutoff[0] < arr)&(arr <= cutoff[1]), arr > cutoff[1]]
        choices = [0, 0.5, 1]
        acreq = np.select(conditions, choices)
        return acreq
        
def clean_fk_file(path, fk_file_name_list):
    for root, dics, files in os.walk(path):
        for dic in dics:
            if dic in fk_file_name_list:
                print(f"Successfully remove folder {os.path.join(root, dic)}")
                shutil.rmtree(os.path.join(root, dic))
        for file in files:
            if file in fk_file_name_list:
                print(f"Successfully remove file {os.path.join(root, file)}")
                os.remove(os.path.join(root, file))

def clean_dataset_filetype(dataset_path, type_list):
    for root, dics, files in os.walk(dataset_path):
        for file in files:
            if not file.endswith(type_list):
                os.remove(os.path.join(root, file))
                print(f"{os.path.join(root, file)} is not a {type_list} file, deleted")
                               
def clean_useless_image(path):
    for root, dics, files in os.walk(path):
        for file in files:
            if file.endswith(('png', 'jpg', 'jpeg')):
                if os.path.getsize(os.path.join(root, file)) == 0:
                    os.remove(os.path.join(root, file))
                    print(f"{os.path.join(root, file)} is invalid image, deleted")

def check_dic_existence(path):
    if not os.path.exists(path):
        os.makedirs(path)
    
def split_dataset(target_path, ratio = (0.7, 0.15, 0.15)):
    """split dataset for training and validation
    Expected target dictionary tree:
    - TaskName
        - 0-Class_i
            - example1.png
            - example2.png
            - example3.png
            ...
        - 1-Class_j
        - ...
    Output dictionary tree has the same structure, e.g.,
    - TaskName/Train
        - 0-Class_i
            - example1.png
            - example2.png
            - example3.png
            ...
        - 1-Class_j
        - ...
    """
    assert sum(ratio) == 1 and len(ratio) == 3, "sum ratio should contain 3 entries with sum = 1"
    clean_fk_file(target_path, '.DS_Store')
    clean_fk_file(target_path, '.ipynb_checkpoints')
    clean_useless_image(target_path)
    
    train_path = os.path.join(target_path, 'Train')
    valid_path = os.path.join(target_path, 'Valid')
    test_path = os.path.join(target_path, 'Test')
    
    for p in [train_path, valid_path, test_path]:
        if os.path.exists(p):
            shutil.rmtree(p)
            
    for cla in os.listdir(target_path):
        sub_path = os.path.join(target_path, cla)
        file_list = np.asarray(os.listdir(sub_path)
)
        permutation = np.random.permutation(len(file_list))
        index1 = int(len(file_list) * ratio[0])
        index2 = int(len(file_list) * (ratio[0] + ratio[1]))            

        train_files = file_list[permutation[ : index1]]
        valid_files = file_list[permutation[index1 : index2]]
        test_files = file_list[permutation[index2 : ]]
        
        cla_train_path = os.path.join(train_path, cla)
        cla_valid_path = os.path.join(valid_path, cla)
        cla_test_path = os.path.join(test_path, cla)
        
        for p, files in zip([cla_train_path, cla_valid_path, cla_test_path], [train_files, valid_files, test_files]):
            os.makedirs(p)
            [shutil.copy(os.path.join(sub_path, i), p) for i in files]
        print(f"split {sub_path} to {cla_train_path} is ready")
    print("All done!")

def compose_functions(*functions):
    return functools.reduce(lambda f, g: lambda x: g(f(x)), functions, lambda x: x)

def count_file(path):
    count = 0
    for root, _, files in os.walk(path):
        for file in files:
            count += 1
    return count

class EasyDict(dict):
    """Convenience class that behaves exactly like dict(), but allows accessing the keys and values using the attribute syntax, i.e., "mydict.key = value".
    """
    def __init__(self, *args, **kwargs): super().__init__(*args, **kwargs)
    def __getattr__(self, name): return self[name]
    def __setattr__(self, name, value): self[name] = value
    def __delattr__(self, name): del self[name]
        
def deprocess_image(x):
    '''
    Same normalization as in:
    https://github.com/fchollet/keras/blob/master/examples/conv_filter_visualization.py
    '''
    if np.ndim(x) > 3:
        x = np.squeeze(x)
    # normalize tensor: center on 0., ensure std is 0.1
    x -= x.mean()
    x /= (x.std() + 1e-5)
    x *= 0.1

    # clip to [0, 1]
    x += 0.5
    x = np.clip(x, 0, 1)

    # convert to RGB array
    x *= 255
    if K.image_dim_ordering() == 'th':
        x = x.transpose((1, 2, 0))
    x = np.clip(x, 0, 255).astype('uint8')
    return x

def adjust_dynamic_range(data, drange_in, drange_out):
    if drange_in != drange_out:
        scale = (np.float32(drange_out[1]) - np.float32(drange_out[0])) / (np.float32(drange_in[1]) - np.float32(drange_in[0]))
        bias = (np.float32(drange_out[0]) - np.float32(drange_in[0]) * scale)
        data = data * scale + bias
    return data