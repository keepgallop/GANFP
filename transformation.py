from scipy import fftpack
import numpy as np
import PIL

def rgb_to_ycc(arr):
    assert isinstance(arr, np.ndarray) and arr.ndim == 3
    arr = PIL.Image.fromarray(arr.astype(np.uint8))
    arr = arr.convert('YCbCr')
    return np.asarray(arr)   

def dct2(arr):
    arr = fftpack.dct(arr, type=2, norm="ortho", axis=0)
    arr = fftpack.dct(arr, type=2, norm="ortho", axis=1)
    return arr

def idct2(arr):
    arr = fftpack.idct(arr, type=2, norm="ortho", axis=0)
    arr = fftpack.idct(arr, type=2, norm="ortho", axis=1)
    return arr

def log_scale(arr, epsilon=1e-12):
    """Log scale the input array.
    """
    arr = np.abs(arr)
    arr += epsilon  # no zero in log
    return 20*np.log(arr)

def rescale(arr, rescale_factor):
    assert isinstance(arr, np.ndarray)
    return arr * (1. / rescale_factor)

def normalize(arr, normalize_mode, normalize_factor = None, color_mode = 'rgb'):
    """normalize_factor: (mean, std) of a batch
    """
    assert isinstance(arr, np.ndarray)
    assert color_mode in ['rgb', 'gray', 'ycc']
    if color_mode == 'gray':
        if normalize_mode in ['samplewise max-min', 'smm']:
            return (arr - np.min(arr)) / (np.max(arr) - np.min(arr))
        elif normalize_mode in ['samplewise z-score', 'sz']:
            return (arr - np.mean(arr)) / (np.std(arr))
        elif normalize_mode in ['featurewise max-min', 'fmm']:
            assert len(normalize_factor) == 2
            return (arr - normalize_factor[1]) / (normalize_factor[0] - normalize_factor[1])
        elif normalize_mode in ['featurewise z-score', 'fz']:
            assert len(normalize_factor) == 2
            return (arr - normalize_factor[1]) / normalize_factor[0]
    
    else:
        def _max_min(normalize_factor, new_arr):
            myzip = zip(normalize_factor[0], normalize_factor[1], new_arr)
            return np.asarray([(z[2]-z[1])/(z[0]-z[1]) for z in myzip])
        def _z_score(normalize_factor, new_arr):
            myzip = zip(normalize_factor[0], normalize_factor[1], new_arr)
            return np.asarray([(z[2]-z[0])/(z[1]) for z in myzip])

        t_axis = (0,1)
        new_arr = [arr[:,:,0], arr[:,:,1], arr[:,:,2]]
        
        if normalize_mode in ['samplewise max-min', 'smm']:
            normalize_factor = (np.max(arr, t_axis), np.min(arr, t_axis))
            return np.transpose(_max_min(normalize_factor, new_arr), (1,2,0))
        elif normalize_mode in ['samplewise z-score', 'sz']:
            normalize_factor = (np.mean(arr, t_axis), np.std(arr, t_axis))
            return np.transpose(_z_score(normalize_factor, new_arr), (1,2,0))
        elif normalize_mode in ['featurewise max-min', 'fmm']:
            assert len(normalize_factor[0]) == 3 and len(normalize_factor[1]) == 3 
            return np.transpose(_max_min(normalize_factor, new_arr), (1,2,0))
        elif normalize_mode in ['featurewise z-score', 'fz']:
            assert len(normalize_factor[0]) == 3 and len(normalize_factor[1]) == 3 
            return np.transpose(_z_score(normalize_factor, new_arr), (1,2,0))

