import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter(action='ignore', category=FutureWarning)


import argparse
import functools
import os

import joblib
# import tensorflow as tf
import keras
from keras.models import load_model
import matplotlib.pyplot as plt
import pandas as pd
import shutil
import image_generator
import nets
import transformation as T
import utils
import numpy as np
from evaluation.normal_stats import unify_pred_true_y, confusion, accuracy
from visualization import DeconvNet
import time
import gc
from keras.utils import to_categorical


def preprocessing_pipeline(dataset_name, color_mode, feature_space, rescale, logscale, normalize_mode, conf_file='./normalize_conf.conf'):
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
    current_args = f"{dataset_name}-{color_mode}-{feature_space}-r{rescale}-l{logscale}-"
    # 3. normalize
    if normalize_mode == 'none':
        NORMALIZE = placeholder_func
    elif normalize_mode in ['smm', 'sz']:
        NORMALIZE = functools.partial(T.normalize,
                                      normalize_mode=normalize_mode,
                                      normalize_factor=None,
                                      color_mode=color_mode)
    else:
        assert os.path.exists(conf_file), f"Cannot found {conf_file}"
        conf_dic = joblib.load('./normalize_conf.conf')

        for i, j in enumerate(conf_dic['args']):
            if current_args == j:
                n_mean, n_std, n_max, n_min = conf_dic['normalize_factor'][i]
                break
        else:
            raise OSError("Cannot find normalize factors for current configs")

        if normalize_mode == 'fmm':
            NORMALIZE = functools.partial(T.normalize,
                                          normalize_mode=normalize_mode,
                                          normalize_factor=[n_max, n_min],
                                          color_mode=color_mode)
        else:
            NORMALIZE = functools.partial(T.normalize,
                                          normalize_mode=normalize_mode,
                                          normalize_factor=[n_mean, n_std],
                                          color_mode=color_mode)

    # 4. compose
    funcs_pipeline = utils.compose_functions(
        COLOR_TRANS, SPACE_TRANS, SCALE, NORMALIZE)

    return funcs_pipeline


def train_rep_model(model=None,
                    epochs=1000,
                    train_data=0,
                    valid_data=0,
                    steps_per_epoch=0,
                    validation_steps=0,
                    model_path=0,
                    his_path=0,
                    earlystop=True):

    modelcheckpoint_callbacks = keras.callbacks.ModelCheckpoint(
        filepath=model_path)
    if earlystop:
        earlystop_callbacks = keras.callbacks.EarlyStopping(monitor="val_loss",
                                                            patience=3,
                                                            verbose=0,
                                                            restore_best_weights=False)
        my_callbacks = [modelcheckpoint_callbacks, earlystop_callbacks]
    else:
        my_callbacks = [modelcheckpoint_callbacks]

#     history = model.fit_generator(x=train_data,
#                         epochs=epochs,
#                         callbacks=my_callbacks,
#                         validation_data=valid_data,
#                         steps_per_epoch=steps_per_epoch,
#                         validation_steps=validation_steps)
    history = model.fit_generator(train_data,
                                  epochs=epochs,
                                  callbacks=my_callbacks,
                                  validation_data=valid_data,
                                  steps_per_epoch=steps_per_epoch,
                                  validation_steps=validation_steps)
    joblib.dump(history.history, his_path)


def plot_training_log(log_paths=None, result_path=None):
    """plot training logs
    Args: log_paths: a list or str of end log. Can be a return value of os.listdir()
    """
    if isinstance(log_paths, str):
        log_paths = [log_paths]
    log_paths.sort()

    fig = plt.figure(figsize=(20, 4))
    ax1 = fig.add_subplot(1, 4, 1)
    ax2 = fig.add_subplot(1, 4, 2)
    ax3 = fig.add_subplot(1, 4, 3)
    ax4 = fig.add_subplot(1, 4, 4)

    def _parser_log_path(log_path):
        dataset_name, space_name, feature_name = log_path.split(
            "/")[-1].split("-")[0:3]
        if space_name == 'ycc':
            space_name = 'YCbCr'
        elif space_name == 'rgb':
            space_name = "RGB"

        if feature_name == 'dct':
            feature_name = 'DCT'
        else:
            feature_name = 'Raw'
        return dataset_name, space_name, feature_name
    legend = []

    for i, log_path in enumerate(log_paths):
        dataset, space, feature = _parser_log_path(log_path)
        log = joblib.load(log_path)
        acc = log["acc"]
        loss = log["loss"]
        val_acc = log["val_acc"]
        val_loss = log["val_loss"]
        ax1.plot(acc, linewidth=1.5)
        ax2.plot(loss, linewidth=1.5)
        ax3.plot(val_acc, linewidth=1.5)
        ax4.plot(val_loss, linewidth=1.5)
        legend.append(space + "-" + feature)
        [[ax.grid('off'), ax.legend(legend)] for ax in [ax1, ax2, ax3, ax4]]

    if len(log_paths) == 1:
        save_path = os.path.join(
            result_path, f'training_logs_{dataset}-{space}-{feature}.eps')
    else:
        save_path = os.path.join(result_path, f'training_logs_{dataset}.eps')
    fig.savefig(os.path.join(save_path), format='eps')
    print(f"Log analysis done. Saved to {save_path}")

def _parser_model_path(p):
        dataset_name, color_mode, feature_space, rescale, logscale, normalize_mode = p.split(
            '-')[0:6]
        rescale, logscale = list(
            map(lambda x: True if "True" in x else False, [rescale, logscale]))
        return dataset_name, color_mode, feature_space, rescale, logscale, normalize_mode
    
def eval_rep_model(model_path=None, test_data_path=0, result_path=0):
    """evaluate model on test data (should be a generator with the same preprocess function as "train_data")
    """
    
    eval_result = {"model_name": [],
                   "test_data_path": [],
                   "results_pred": [],
                   "results_true": [],
                   "label":[]}
    eval_result['test_data_path'].append(test_data_path)
    
    for p in os.listdir(model_path):
        dataset_name, color_mode, feature_space, rescale, logscale, normalize_mode = _parser_model_path(p)
        funcs_pipeline = preprocessing_pipeline(dataset_name,
                                                color_mode,
                                                feature_space,
                                                rescale,
                                                logscale,
                                                normalize_mode)
        
        test_data_gen_paras = {'path': test_data_path,
                               'preprocessing_function': funcs_pipeline,
                               'batch_size': 128,
                               'target_size': (128, 128),
                               'color_mode': color_mode,
                               'shuffle': False}
        
        test_generator = image_generator.data_gen(**test_data_gen_paras)
        count_test = utils.count_file(test_data_path)
        model = load_model(os.path.join(model_path, p))
        pred = model.predict_generator(test_generator,
                                       verbose=1,
                                       steps=count_test//test_data_gen_paras['batch_size'] + 1,)

        eval_result['model_name'].append(p)
        eval_result['results_pred'].append(pred)
        eval_result['results_true'].append(test_generator.classes)
        eval_result["label"].append(test_generator.class_indices)
        
    save_path = os.path.join(result_path, dataset_name + "_eval_result.pkl")
    joblib.dump(eval_result, save_path)
    print(f"Save prediction results to {save_path}")
    return eval_result, save_path

def analyze_results(eval_result, save_path, result_path):
    """Statistically analyze evaluation results outputed by eval_rep_model()
    """
    eval_result['acc'] = []
    eval_result['cm'] = []
    for i, model_name in enumerate(eval_result['model_name']):
        y_pred = eval_result['results_pred'][i]
        y_true = eval_result['results_true'][i]
        y_true, y_pred = unify_pred_true_y(y_true, y_pred)
        
        # plot confusion matrix
        cm_save_path = os.path.join(result_path, model_name + "_confusion_matrix.eps")
        labels = [i.split('-')[-1] for i in eval_result["label"][0]]
        cm = confusion(y_true, y_pred, labels = labels, save_path=cm_save_path)
        acc = accuracy(y_true, y_pred)
        eval_result['acc'].append(acc)
        eval_result['cm'].append(cm)
    joblib.dump(eval_result, save_path)
    print(f"Update prediction results to {save_path}")
    return eval_result, save_path  



def vis_rep_model_per_img(model_path=None, im_path=None, result_path=0, savefig=False, fig_save_path=0, target_count = 256):
    """model_path = model full path
    """
    save_path = os.path.join(result_path, "fingerprints_per_img")
    utils.check_dic_existence(save_path)
#     print(f"------ Path to save fingerprint: {save_path} -----")
    model_name = model_path.split('/')[-1]
    model = load_model(model_path)

    dataset_name, color_mode, feature_space, rescale, logscale, normalize_mode = run._parser_model_path(model_name)
    funcs_pipeline = run.preprocessing_pipeline(dataset_name,
                                            color_mode,
                                            feature_space,
                                            rescale,
                                            logscale,
                                            normalize_mode)
    
    if  type(im_path) == list:
        im_list = im_path
    else:
        im_list = [os.path.join(im_path, i) for i in os.listdir(im_path)]
        
    for count, im_name in enumerate(im_list):
        results_dic = {'model':model_name,
               'model_results':[],}
        
        if im_name.endswith('png'):
            im_path_single = im_name
        im_raw = utils.load_img_arr_128(im_path_single)
        im_pre = funcs_pipeline(im_raw)

        input_data=im_pre[np.newaxis, :]
        preds = model.predict(input_data)
        selected_unit = np.zeros(preds.shape)
        selected_unit[0, np.argmax(preds)] = 1.
        GB = vis.GuidedBackprop(model=model,
                              layer_name=model.layers[-1].name,
                              input_data=input_data,
                              masking=selected_unit)

        im_f_mask = GB.compute()

        cam_image, heat_map = vis.grad_cam(model, input_data, np.argmax(preds), "conv2d_4")
        grad_cam_img = im_f_mask * heat_map[..., np.newaxis]
        fingerprint  = utils.deprocess_image(grad_cam_img)

#         im_normalized_f_mask = T.normalize(im_f_mask, 'smm', color_mode)

#         im_fingerprint = np.multiply(im_pre, im_normalized_f_mask)
#         im_fingerprint_with_round_mask = np.multiply(im_pre, utils.round_thre(im_normalized_f_mask))
#         im_fingerprint_with_cutoff_mask = np.multiply(im_pre, utils.cutoff_thre(im_normalized_f_mask, 0.5))

        results = {}
        results['im_raw'] = im_raw
        results['pred'] = np.argmax(preds)
        results['im_pre'] = im_pre
        results['im_f_mask'] = im_f_mask
        results['heat_map'] = heat_map
        results['fingerprint'] = fingerprint
        results_dic['model_results'].append(results)

        joblib.dump(results_dic, os.path.join(save_path, im_path_single.split('/')[-1].split('.')[0] + '.h5'))
        if savefig:
            imsave(fig_save_path + 'fp-' + im_path_single.split('/')[-1], fingerprint)
#         print('Done!', im_path_single.split('/')[-1].split('.')[0], np.argmax(preds))
        print(f"{count}/{target_count}, pred={np.argmax(preds)}", end = '\r')
                
########################################
#####  below are working funtions  #####
########################################
def start_training_workflow(args):
    print("******************* Start training ********************")
    paraDic = utils.EasyDict()

    print("----Cleaning workspace----")
    utils.clean_fk_file('.', '.DS_Store')
    utils.clean_fk_file('.', '.ipynb_checkpoints')
    utils.clean_useless_image('.')
    print("Done!")

    print("----Loading model----")
    # parameters for representation network
    paraDic.rep_model_paras = {'weights': None,
                               'class_num': 5,
                               'lr': 0.001, }
    if args.color_mode == 'gray':
        paraDic.rep_model_paras['input_shape'] = (128, 128, 1)
    else:
        paraDic.rep_model_paras['input_shape'] = (128, 128, 3)

    RepresentationModels = nets.RepresentationModels(
        **paraDic.rep_model_paras)
    model = RepresentationModels.get_CNN()
    print("Done!")

    print("----Loading data----")
    # parameters for data generator
    # preprocessing pipeline
    funcs_pipeline = preprocessing_pipeline(args.dataset_name,
                                            args.color_mode,
                                            args.feature_space,
                                            args.rescale,
                                            args.logscale,
                                            args.normalize_mode)

    paraDic.train_data_gen_paras = {'path': args.train_dataset_path,
                                    'preprocessing_function': funcs_pipeline,
                                    'batch_size': 128,
                                    'target_size': (128, 128),
                                    'color_mode': args.color_mode,
                                    'shuffle': True}

    paraDic.valid_data_gen_paras = paraDic.train_data_gen_paras.copy()
    paraDic.valid_data_gen_paras.update({'path': args.valid_dataset_path,
                                         'batch_size': 128})
    print("Training generator summary")
    train_generator = image_generator.data_gen(**paraDic.train_data_gen_paras)
    print("Validation generator summary")
    valid_generator = image_generator.data_gen(**paraDic.valid_data_gen_paras)
    print("Done!")

    print("----Training model----")
    #  parameters for model training
    count_t = utils.count_file(args.train_dataset_path)
    count_v = utils.count_file(args.valid_dataset_path)
    current_args = f"{args.dataset_name}-{args.color_mode}-{args.feature_space}-r{args.rescale}-l{args.logscale}-{args.normalize_mode}-"
    paraDic.training_para = {'epochs': 200,
                             'steps_per_epoch': count_t // paraDic.train_data_gen_paras['batch_size'],
                             'validation_steps': count_v // paraDic.valid_data_gen_paras['batch_size'],
                             'model_path': os.path.join(args.model_path, current_args + 'model.{epoch:02d}-{val_loss:.2f}.h5'),
                             'his_path': os.path.join(args.his_path, current_args + 'log.pkl'),
                             'earlystop': True}

    train_rep_model(model=model,
                    train_data=train_generator,
                    valid_data=valid_generator,
                    **paraDic.training_para)
    print("Done!")
    print(f"model files save to {paraDic.training_para['his_path']}")
    print("******************* End training ********************")

    
def start_evaluation_workflow(args):
    print("******************* Start evaluation ********************")
    print("----Cleaning workspace----")
    utils.clean_fk_file('.', '.DS_Store')
    utils.clean_fk_file('.', '.ipynb_checkpoints')
    utils.clean_useless_image('.')
    print("Done!")
    
    print("----Ploting training logs----")
    if args.plot_training_log:
        if os.path.isdir(args.log_path):
            log_paths = [os.path.join(args.log_path, i)
                         for i in os.listdir(args.log_path)]
        else:
            log_paths = args.log_path
        plot_training_log(log_paths, args.result_path)
    print("Done!")

    print("----Evaluating models----")
    eval_result, save_path = eval_rep_model(args.model_path, args.test_dataset_path, args.result_path)
    print("Done!")

    print("----Analyzing results----")
    eval_result, save_path = analyze_results(eval_result, save_path, args.result_path)
    print("Done!")
    print("******************* End evaluation ********************")


def start_visualization_workflow(args):
    print("******************* Start evaluation ********************")
    print("----Cleaning workspace----")
    utils.clean_fk_file('.', '.DS_Store')
    utils.clean_fk_file('.', '.ipynb_checkpoints')
    utils.clean_useless_image('.')
    print("Done!")
    
    print("----Visualize models----")
    for i in os.listdir(args.vis_dataset_path):
        im_list = []
        if  i != '.DS_Store':
            print('===========>', i)
            sub_path  = os.path.join(args.vis_dataset_path, i)
            for file in os.listdir(sub_path):
                if file.endswith('png'):
                    im_list.append(os.path.join(sub_path, file))
                if len(im_list) == args.num_per_class:
                    break
    vis_rep_model_per_img(args.model_path, im_list, args.result_path, target_count=args.num_per_class, savefig=False)
    print("Done!")

    print("----Stats-based results----")

    print("Done!")
    print("******************* End evaluation ********************")
    
    pass


########################################
#####  above are working funtions  #####
########################################

def argparser():
    parser = argparse.ArgumentParser()
    app = parser.add_subparsers(
        help="Application {training|evluation|visualization}.", dest="app")

    # args for training mode
    parser_t = app.add_parser("training")

    parser_t.add_argument('--dataset', '-d',
                          default='CelebA', dest='dataset_name', help="Used as identifier to specify file names")
    parser_t.add_argument('--color_mode', '-c', default='rgb',
                          dest='color_mode', help="Supported modes rgb/gray/ycc")
    parser_t.add_argument('--feature_space', '-f',
                          default='raw', dest='feature_space', help="Supported space dct/raw")

    parser_t.add_argument(
        '--rescale', '-r', dest='rescale', action='store_true')
    parser_t.add_argument('--logscale', '-l',
                          dest='logscale', action='store_true')
    parser_t.add_argument('--normalize_mode', '-n',
                          default='fz', dest='normalize_mode', help="Supported modes none/smm/sz/fmm/fz")

    parser_t.add_argument('--modelpath', '-mp',
                          default='./model/', dest='model_path')
    parser_t.add_argument('--logpath', '-lp',
                          default='./logs/', dest='his_path')
    parser_t.add_argument('--traindatasetpath', '-tp',
                          dest='train_dataset_path')
    parser_t.add_argument('--validdatasetpath', '-vp',
                          dest='valid_dataset_path')

    # args for evaluation mode
    parser_e = app.add_parser("evaluation")
    
    parser_e.add_argument(
        '--plot_training_log', '-p', dest='plot_training_log', action='store_true')
    parser_e.add_argument(
        '--logpath', '-lp', default='./logs/', dest='log_path')
    parser_e.add_argument('--result_path', '-rp',
                          default='./results/', dest='result_path')

    parser_e.add_argument('--model_path', '-mp',
                          default='./models/', dest='model_path')
    parser_e.add_argument('--test_dataset_path', '-tp', default='./results/', dest='test_dataset_path')
    
    
    # args for visualization mode
    parser_v = app.add_parser("visualization")
    
    parser_v.add_argument('--result_path', '-rp',
                          default='./results/', dest='result_path')

    parser_v.add_argument('--model_path', '-mp',
                          default='./models/', dest='model_path')
    parser_v.add_argument('--vis_dataset_path', '-tp', default='./results/', dest='vis_dataset_path')
    parser_v.add_argument('--num_per_class', '-n', type=int, default=100, dest='num_per_class')
    
    
    args = parser.parse_args()
    return args

#


def main():
    args = argparser()
    assert args.app in ['training', 'evaluation', 'visualization']

    if args.app == 'training':
        # check args
        utils.check_dic_existence(args.model_path)
        utils.check_dic_existence(args.his_path)

        assert args.color_mode in ['gray', 'rgb', 'ycc']
        assert args.feature_space in ['raw', 'dct']
        assert args.normalize_mode in ['none', 'smm', 'sz', 'fmm', 'fz']

        # start workflow
        start_training_workflow(args)

    elif args.app == 'evaluation':
        # check args
        utils.check_dic_existence(args.result_path)
        
        # start workflow
        start_evaluation_workflow(args)

    elif args.app == 'visualization':
        # check args
        utils.check_dic_existence(args.result_path)
        
        start_visualization_workflow(args)
        pass


if "__main__" == __name__:
    main()
