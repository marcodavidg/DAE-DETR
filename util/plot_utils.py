"""
Plotting utilities to visualize training logs.
"""
import torch
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
import pdb
import os

from pathlib import Path, PurePath


def plot_logs(logs, fields=('class_error', 'loss_bbox_unscaled', 'mAP'), ewm_col=0, log_name='log.txt'):
    '''
    Function to plot specific fields from training log(s). Plots both training and test results.

    :: Inputs - logs = list containing Path objects, each pointing to individual dir with a log file
              - fields = which results to plot from each log file - plots both training and test for each field.
              - ewm_col = optional, which column to use as the exponential weighted smoothing of the plots
              - log_name = optional, name of log file if different than default 'log.txt'.

    :: Outputs - matplotlib plots of results in fields, color coded for each log file.
               - solid lines are training results, dashed lines are test results.

    '''
    func_name = "plot_utils.py::plot_logs"

    # verify logs is a list of Paths (list[Paths]) or single Pathlib object Path,
    # convert single Path to list to avoid 'not iterable' error

    if not isinstance(logs, list):
        if isinstance(logs, PurePath):
            logs = [logs]
            print(f"{func_name} info: logs param expects a list argument, converted to list[Path].")
        else:
            raise ValueError(f"{func_name} - invalid argument for logs parameter.\n \
            Expect list[Path] or single Path obj, received {type(logs)}")

    # Quality checks - verify valid dir(s), that every item in list is Path object, and that log_name exists in each dir
    for i, dir in enumerate(logs):
        if not isinstance(dir, PurePath):
            raise ValueError(f"{func_name} - non-Path object in logs argument of {type(dir)}: \n{dir}")
        if not dir.exists():
            raise ValueError(f"{func_name} - invalid directory in logs argument:\n{dir}")
        # verify log_name exists
        fn = Path(dir / log_name)
        if not fn.exists():
            print(f"-> missing {log_name}.  Have you gotten to Epoch 1 in training?")
            print(f"--> full path of missing log file: {fn}")
            return

    # load log file(s) and plot
    dfs = [pd.read_json(Path(p) / log_name, lines=True) for p in logs]
    maxAP = 0
    lastAP = 0
    currentEpoch = -1
    fig, axs = plt.subplots(ncols=len(fields), figsize=(16, 5))
    for df, color in zip(dfs, sns.color_palette(n_colors=len(logs))):
        currentEpoch += 1
        for j, field in enumerate(fields):
            if field == 'mAP':
                coco_eval = pd.DataFrame(
                    np.stack(df.test_coco_eval_bbox.dropna().values)[:, 1]
                ).ewm(com=ewm_col).mean()
                axs[j].plot(coco_eval, c=color)
                lastAP = 0#coco_eval[0][49]
                sorted_out = coco_eval.sort_values(by=0, ascending=False).reset_index(drop=False)
                print(sorted_out, lastAP)
                # print(coco_eval)
            else:
                #pdb.set_trace()
                try:
                    df = df.drop(['now_time','epoch_time'], axis=1)
                except:
                    pass
                df.interpolate().ewm(com=ewm_col).mean().plot(
                    y=[f'train_{field}', f'test_{field}'],
                    ax=axs[j],
                    color=[color] * 2,
                    style=['-', '--']
                )
    for ax, field in zip(axs, fields):
        if field == 'mAP':
            ax.legend([Path(p).name for p in logs])
            # breakpoint()
            ax.set_title(field + " " + str(sorted_out[0][0]) + " " + str(lastAP))
        else:
            ax.legend([f'train', f'test'])
            ax.set_title(field)

#    pdb.set_trace()
    return fig, axs

def plot_precision_recall(files, naming_scheme='iter'):
    # if naming_scheme == 'exp_id':
    #     # name becomes exp_id
    #     names = [f.parts[-3] for f in files]
    # elif naming_scheme == 'iter':
    #     names = [f.stem for f in files]
    # else:
    #     raise ValueError(f'not supported {naming_scheme}')
    # fig, axs = plt.subplots(ncols=2, figsize=(16, 5))
    # for f, color, name in zip(files, sns.color_palette("Blues", n_colors=len(files)), names):
    data = torch.load('../exps/HD256_AuxNetwork_C2F_source/checkpoint.pth')
    # precision is n_iou, n_points, n_cat, n_area, max_det
    breakpoint()
    precision = data['precision']
    recall = data['params'].recThrs
    scores = data['scores']
    # take precision for all classes, all areas and 100 detections
    precision = precision[0, :, :, 0, -1].mean(1)
    scores = scores[0, :, :, 0, -1].mean(1)
    prec = precision.mean()
    rec = data['recall'][0, :, 0, -1].mean()
    print(f'{naming_scheme} {name}: mAP@50={prec * 100: 05.1f}, ' +
          f'score={scores.mean():0.3f}, ' +
          f'f1={2 * prec * rec / (prec + rec + 1e-8):0.3f}'
          )
        # axs[0].plot(recall, precision, c=color)
        # axs[1].plot(recall, scores, c=color)

    # axs[0].set_title('Precision / Recall')
    # axs[0].legend(names)
    # axs[1].set_title('Scores / Recall')
    # axs[1].legend(names)
    # return fig, axs
    return None




# def load_eval(eval_path):
#     data = torch.load(eval_path)
#     # precision is n_iou, n_points, n_cat, n_area, max_det
#     precision = data['precision']
#     # take precision for all classes, all areas and 100 detections
#     CLASSES = [
#         'N/A', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
#         'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
#         'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
#         'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
#         'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
#         'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
#         'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
#         'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
#         'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
#         'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
#         'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
#         'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
#     ]
#     CLASSES = [c for c in CLASSES if c != 'N/A']
#     area = 0
#     return pd.DataFrame.from_dict({c: p for c, p in zip(CLASSES, precision[0, :, :, area, -1].mean(0) * 100)}, orient='index')
#
# with pd.option_context('display.max_rows', None, 'display.max_columns', None):
#     df = load_eval('../exps/HD256_AuxNetwork_C2F_source/checkpoint.pth')
#     print(df)
# #
# #
# #
#
#
# for file_name in os.listdir("./exps/HD512_NewFusionModuleCANNY_NormalEpochs_DA/"):
file_names = ["HD256_AuxNetwork_C2F_da"]
# printPlot = False
printPlot = True


def load_eval(eval_path):
    data = torch.load(eval_path)
    # precision is n_iou, n_points, n_cat, n_area, max_det
    precision = data['precision']
    # take precision for all classes, all areas and 100 detections
    CLASSES = [
        'N/A', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
        'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
        'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
        'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
        'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
        'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
        'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
        'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
        'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
        'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
        'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
        'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
    ]
    CLASSES = [c for c in CLASSES if c != 'N/A']
    area = 0
    return pd.DataFrame.from_dict({c: p for c, p in zip(CLASSES, precision[0, :, :, area, -1].mean(0) * 100)}, orient='index')

with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    df = load_eval('../exps/eval_HD256_AuxNetwork_C2F_da/eval.pth')
    print(df)

#
# with pd.option_context('display.max_rows', None, 'display.max_columns', None):
#     df = load_eval('../exps/eval2_HD256_AuxNetwork_C2F_source/eval.pth')
#     print(df)


# for file_name in file_names:
#     files = (Path(f'../exps/{file_name}'))
#
#     fig, _ = plot_logs(files)
#     # fig.show()
#     if printPlot:
#         fig.savefig(f'../plots/{file_name}.png')
#     print("Success...", file_name)