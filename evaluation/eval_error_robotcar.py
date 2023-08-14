import argparse
import json
import os
import os.path as osp
import sys

import cv2
import numpy as np

sys.path.append('..')

from tqdm import trange, tqdm
from ui import PyTable
from utils import read_list_from_file
from transforms import CenterCrop
from datasets import ROBOTCAR_ROOT
from matplotlib import cm, pyplot as plt

# target size
_TARGET_SIZE = (1280, 640)


def draw_scatter_image(image, points, cmap="turbo"):
    points = cv2.resize(points, (image.shape[1], image.shape[0]))
    normalized_matrix = cv2.normalize(points, None, 0, 255, cv2.NORM_MINMAX)
    normalized_matrix = np.uint8(normalized_matrix)

    _, mask_err = cv2.threshold(normalized_matrix, 0, 255, cv2.THRESH_BINARY)

    im = image.copy()
    cmap = plt.get_cmap(cmap)
    rmse_error_colored = cmap(points)

    # Draw scatter points on the image using the mask
    x, y = np.where(mask_err)
    color = rmse_error_colored[x, y]

    for i in range(len(x)):
        color_ = tuple((255*color[i]).astype(int))[:3]
        cv2.circle(im, (y[i], x[i]), 1, (int(color_[0]),int(color_[1]),int(color_[2])) , -1)
    return im

def compute_metrics(pred, gt):
    thresh = np.maximum((gt / pred), (pred / gt))
    a1 = (thresh < 1.25).mean()
    a2 = (thresh < 1.25 ** 2).mean()
    a3 = (thresh < 1.25 ** 3).mean()

    rmse = (gt - pred) ** 2
    rmse = np.sqrt(rmse.mean())
    rmse_log = (np.log(gt) - np.log(pred)) ** 2
    rmse_log = np.sqrt(rmse_log.mean())

    abs_rel = np.mean(np.abs(gt - pred) / gt)
    sq_rel = np.mean(((gt - pred) ** 2) / gt)

    result = {'abs_rel': abs_rel, 'sq_rel': sq_rel, 'rmse': rmse, 'rmse_log': rmse_log, 'a1': a1, 'a2': a2, 'a3': a3}
    return result


def read_gt(dn, fs):
    """
    Read ground truth from given directory
    :param dn: directory name
    :param fs: files
    :return:
    """
    result = []
    for f in fs:
        # result.append(np.load(os.path.join(dn, '{}_depth.npy'.format(f))))
        result.append(cv2.imread(os.path.join(dn, '{}.png'.format(f)))[:,:,1])

    return result

def read_images(dn, fs):
    result = []
    for f in fs:
        path = os.path.join(dn, '{}.png'.format(f))
        print(path)
        bgr = cv2.imread(path)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        result.append(rgb) 
    return result 

def print_table(title, data, str_format):
    table = PyTable(list(data.keys()), title)
    table.add_item({k: str_format.format(v) for k, v in data.items()})
    table.print_table()


def evaluate():
    # check length
    pred_len, gt_len = len(pred_depth), len(gt_depth)
    assert pred_len == gt_len, 'The length of predictions must be same as ground truth.'
    # store result
    errors = {'abs_rel': [], 'sq_rel': [], 'rmse': [], 'rmse_log': [], 'a1': [], 'a2': [], 'a3': []}
    # transform
    crop = CenterCrop(*_TARGET_SIZE)
    # compute loss
    for i in trange(gt_len):
        # get item
        pred, gt, img = pred_depth[i], gt_depth[i], images[i]
        gt = crop(gt, inplace=False)
        img = crop(img, inplace=False)
    
        mask = (gt > args.min_depth) & (gt < args.max_depth)

        # resize
        gt_h, gt_w = gt.shape
        pred = cv2.resize(pred, (gt_w, gt_h), interpolation=cv2.INTER_NEAREST)
        # get values
        pred_vals, gt_vals = pred[mask], gt[mask]

        pred_depth_arr = np.zeros_like(pred)
        gt_depth_arr = np.zeros_like(gt)
        pred_depth_arr[mask] = pred[mask]
        gt_depth_arr[mask] = gt[mask]
        
        if gt_vals.size <= 0:
            raise ValueError('The size of ground truth is zero.')
        # compute scale
        scale = np.median(gt_vals) / np.median(pred_vals)
        pred_vals *= scale
        pred_vals = np.clip(pred_vals, args.min_depth, args.max_depth)

        pred_depth_arr *= scale
        pred_depth_arr = np.clip(pred_depth_arr, args.min_depth, args.max_depth)

        pred_depth_arr_ = np.zeros_like(pred_depth_arr)
        pred_depth_arr_[gt_depth_arr.astype(bool)] = pred_depth_arr[gt_depth_arr.astype(bool)]

        abs_rel_map = np.abs(gt_depth_arr - pred_depth_arr_) / (gt_depth_arr + 1e-5)

        # compute error
        error = compute_metrics(pred_vals, gt_vals)

        abs_rel_map[abs_rel_map < 0.23] = 0
        error_img = draw_scatter_image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), abs_rel_map)

        out_fn = os.path.join("/media/aiteam/DataAI/STEPS/vis/error_map_night", '{}_error.png'.format("%05d" %i))
        cv2.imwrite(out_fn, error_img)

        # add
        for k in errors:
            errors[k].append(error[k])
    # compute mean
    errors = {k: np.mean(v).item() for k, v in errors.items()}
    # done
    tqdm.write('Done.')
    # output
    print_table('Evaluation Result', errors, '{:.4f}')
    # save result
    file_name = args.output_file_name
    if file_name is not None:
        out_dir = osp.dirname(file_name)
        if not os.path.exists(out_dir):
            os.mkdir(out_dir)
        with open(file_name, 'w') as fo:
            json.dump(errors, fo)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('data_root', type=str, help='Root directory of dataset.', choices=['day', 'night'])
    parser.add_argument('--pred_dir', type=str, default='rc_result/', help='Directory where predictions stored.')
    parser.add_argument('--max_depth', type=float, default=60.0, help='Maximum depth value.')
    parser.add_argument('--min_depth', type=float, default=1e-5, help='Minimum depth value.')
    parser.add_argument('--output_file_name', type=str, default=None, help='File path for saving result.')

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    root_dir = ROBOTCAR_ROOT[args.data_root] if args.data_root in ROBOTCAR_ROOT else args.data_root
    files = read_list_from_file(os.path.join(root_dir, 'test_split_adds.txt'), 1)
    gt_depth = read_gt(os.path.join(root_dir, 'depth_adds/'), files)
    pred_depth = np.load(os.path.join(args.pred_dir, 'predictions.npy'))
    images = read_images(os.path.join(root_dir, 'test_adds/'), files)
    evaluate()
