from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from collections import namedtuple
from skimage.metrics import structural_similarity as ssim
from itertools import product
import numpy as np
import random
import os
import cv2
import json
 
Corner = namedtuple('Corner', 'x1 y1 x2 y2')
# alias
BBox = Corner
Center = namedtuple('Center', 'x y w h')

def get_cropped_example(parser=None, gt_bbox=None):
    
    patch_h = gt_bbox[3]
    patch_w = gt_bbox[2]

    candidate_list = candidate_videos(parser)
    for c_index in range(len(candidate_list)):
        dataset_root = os.path.join(parser.data_base_path, parser.dataset)
        candidate_list[c_index] = os.path.join(dataset_root, candidate_list[c_index], 'img')

    for c_index in range(len(candidate_list)):    
        for img_f in os.listdir(candidate_list[c_index]):
            if img_f.split('.')[0][-1] == '1':
                candidate_list[c_index] = os.path.join(candidate_list[c_index], img_f)
                break
    print(candidate_list)
    cropped_list = []
    location_list = []
    candidate_name_list = []
    for crop_path in candidate_list:
        img = cv2.imread(crop_path)
        if img.shape[0] < patch_h or img.shape[1] < patch_w:
            cropped = cv2.resize(img, (patch_w, patch_h), interpolation=cv2.INTER_LINEAR)
            height_start = 0
            height_end = img.shape[0]
            width_start = 0
            width_end = img.shape[1]
        else:
            height_start = np.random.randint(0, img.shape[0]-patch_h+1)
            width_start = np.random.randint(0, img.shape[1]-patch_w+1)
            height_end = height_start+patch_h
            width_end = width_start+patch_w
            cropped = img[height_start:height_end, width_start:width_end]
        cropped_list.append(cropped)
        location_list.append((height_start, height_end, width_start, width_end))
        candidate_name_list.append(crop_path.split('/')[-3]+'_'+crop_path.split('/')[-1].split('.')[0])
    
    attack_path = os.path.join(parser.data_base_path, parser.dataset, parser.video, 'attack_results', parser.model_name)
    used_candidates = os.listdir(attack_path)
    
    candidate_index = np.random.randint(0, len(candidate_list))
    while candidate_name_list[candidate_index] in used_candidates:
        candidate_index = np.random.randint(0, len(candidate_list))
    cropped_example = cropped_list[candidate_index]
    height_start = location_list[candidate_index][0]
    height_end = location_list[candidate_index][1]
    width_start = location_list[candidate_index][2]
    width_end = location_list[candidate_index][3]
    candidate_name = candidate_name_list[candidate_index]
    return height_start, height_end, width_start, width_end, candidate_name, cropped_example

def auc_eval(parser=None):
    return None

def candidate_videos(parser=None):
    candidate_list = []
    with open(os.path.join(parser.data_base_path, parser.dataset, parser.dataset+'.json'), 'r') as f:
        json_content = json.load(f)
        for key in json_content.keys():
            candidate_list.append(key)
    final_candidates = random.sample(candidate_list, parser.num_video)
    print(final_candidates)
    return final_candidates
    
def index_cal(img_height=None, img_width=None, gt_bbox=None, parser=None):
    patch_w = gt_bbox[2] 
    patch_h = gt_bbox[3] 
    
    index_list = []
    for i in range(parser.patch_num):
        x_shift_random = np.random.randint(-min(int(gt_bbox[2]/2), gt_bbox[0]), min(int(gt_bbox[2]/2), img_width-(gt_bbox[0]+gt_bbox[2])))
        index_list.append((gt_bbox[1], gt_bbox[0]+x_shift_random))
    for i in range(parser.patch_num):    
        y_shift_random = np.random.randint(-min(int(gt_bbox[3]/2), gt_bbox[1]), min(int(gt_bbox[3]/2), img_height-(gt_bbox[1]+gt_bbox[3])))
        index_list.append((gt_bbox[1]+y_shift_random, gt_bbox[0]))

    return index_list


def corner2center(corner):
    """ convert (x1, y1, x2, y2) to (cx, cy, w, h)
    Args:
        conrner: Corner or np.array (4*N)
    Return:
        Center or np.array (4 * N)
    """
    if isinstance(corner, Corner):
        x1, y1, x2, y2 = corner
        return Center((x1 + x2) * 0.5, (y1 + y2) * 0.5, (x2 - x1), (y2 - y1))
    else:
        x1, y1, x2, y2 = corner[0], corner[1], corner[2], corner[3]
        x = (x1 + x2) * 0.5
        y = (y1 + y2) * 0.5
        w = x2 - x1
        h = y2 - y1
        return x, y, w, h


def center2corner(center):
    """ convert (cx, cy, w, h) to (x1, y1, x2, y2)
    Args:
        center: Center or np.array (4 * N)
    Return:
        center or np.array (4 * N)
    """
    if isinstance(center, Center):
        x, y, w, h = center
        return Corner(x - w * 0.5, y - h * 0.5, x + w * 0.5, y + h * 0.5)
    else:
        x, y, w, h = center[0], center[1], center[2], center[3]
        x1 = x - w * 0.5
        y1 = y - h * 0.5
        x2 = x + w * 0.5
        y2 = y + h * 0.5
        return x1, y1, x2, y2


def IoU(rect1, rect2):
    """ caculate interection over union
    Args:
        rect1: (x1, y1, x2, y2)
        rect2: (x1, y1, x2, y2)
    Returns:
        iou
    """
    # overlap
    x1, y1, x2, y2 = rect1[0], rect1[1], rect1[2], rect1[3]
    tx1, ty1, tx2, ty2 = rect2[0], rect2[1], rect2[2], rect2[3]

    xx1 = np.maximum(tx1, x1)
    yy1 = np.maximum(ty1, y1)
    xx2 = np.minimum(tx2, x2)
    yy2 = np.minimum(ty2, y2)
    
    ww = np.maximum(0, xx2 - xx1)
    hh = np.maximum(0, yy2 - yy1)

    area = (x2-x1) * (y2-y1)
    target_a = (tx2-tx1) * (ty2 - ty1)
    inter = ww * hh
    iou = inter / (area + target_a - inter)
    return iou

def robust(rect1, rect2):

    #We also need to calculate the robustness of a video clip.
    x1, y1, x2, y2 = rect1[0], rect1[1], rect1[2], rect1[3]
    tx1, ty1, tx2, ty2 = rect2[0], rect2[1], rect2[2], rect2[3]

    xx1 = np.maximum(tx1, x1)
    yy1 = np.maximum(ty1, y1)
    xx2 = np.minimum(tx2, x2)
    yy2 = np.minimum(ty2, y2)
    
    ww = np.maximum(0, xx2 - xx1)
    hh = np.maximum(0, yy2 - yy1)
    
    if (ww * hh) == 0:
        return True
    else:
        return False

def robustness_cal(source_path, target_path):
    source_f = open(source_path, 'r+').readlines()
    target_f = open(target_path, 'r+').readlines()
    count = 0
    for i in range(len(source_f)):
        source = source_f[i][:-1].split(',')
        target = target_f[i][:-1].split(',')
        for s in range(len(source)):
            source[s] = float(source[s])
        for t in range(len(target)):
            target[t] = float(target[t])
        source[2] = source[0] + source[2]
        source[3] = source[1] + source[3]
        target[2] = target[0] + target[2]
        target[3] = target[1] + target[3]
        if robust(source, target):
            count += 1
    return count 

def overlap_cal(source_path, target_path):
    source_f = open(source_path, 'r+').readlines()
    target_f = open(target_path, 'r+').readlines()
    count = 0
    iou_average = 0
    for i in range(len(source_f)):
        source = source_f[i][:-1].split(',')
        target = target_f[i][:-1].split(',')
        for s in range(len(source)):
            source[s] = float(source[s])
        for t in range(len(target)):
            target[t] = float(target[t])
        source[2] = source[0] + source[2]
        source[3] = source[1] + source[3]
        target[2] = target[0] + target[2]
        target[3] = target[1] + target[3]
        iou_average += IoU(source, target)
        count += 1
    #print(iou_average / count)

    return iou_average / count

def cxy_wh_2_rect(pos, sz):
    """ convert (cx, cy, w, h) to (x1, y1, w, h), 0-index
    """
    return np.array([pos[0]-sz[0]/2, pos[1]-sz[1]/2, sz[0], sz[1]])


def rect_2_cxy_wh(rect):
    """ convert (x1, y1, w, h) to (cx, cy, w, h), 0-index
    """
    return np.array([rect[0]+rect[2]/2, rect[1]+rect[3]/2]), \
        np.array([rect[2], rect[3]])


def cxy_wh_2_rect1(pos, sz):
    """ convert (cx, cy, w, h) to (x1, y1, w, h), 1-index
    """
    return np.array([pos[0]-sz[0]/2+1, pos[1]-sz[1]/2+1, sz[0], sz[1]])


def rect1_2_cxy_wh(rect):
    """ convert (x1, y1, w, h) to (cx, cy, w, h), 1-index
    """
    return np.array([rect[0]+rect[2]/2-1, rect[1]+rect[3]/2-1]), \
        np.array([rect[2], rect[3]])


def get_axis_aligned_bbox(region):
    """ convert region to (cx, cy, w, h) that represent by axis aligned box
    """
    nv = region.size
    region = region.astype(float)       
    if nv == 8:
        cx = np.mean(region[0::2])
        cy = np.mean(region[1::2])
        x1 = min(region[0::2])
        x2 = max(region[0::2])
        y1 = min(region[1::2])
        y2 = max(region[1::2])
        A1 = np.linalg.norm(region[0:2] - region[2:4]) * \
            np.linalg.norm(region[2:4] - region[4:6])
        A2 = (x2 - x1) * (y2 - y1)
        s = np.sqrt(A1 / A2)
        w = s * (x2 - x1) + 1
        h = s * (y2 - y1) + 1
    else:
        x = region[0]
        y = region[1]
        w = region[2]
        h = region[3]
        cx = x+w/2
        cy = y+h/2
    return cx, cy, w, h


def get_min_max_bbox(region):
    """ convert region to (cx, cy, w, h) that represent by mim-max box
    """
    nv = region.size
    if nv == 8:
        cx = np.mean(region[0::2])
        cy = np.mean(region[1::2])
        x1 = min(region[0::2])
        x2 = max(region[0::2])
        y1 = min(region[1::2])
        y2 = max(region[1::2])
        w = x2 - x1
        h = y2 - y1
    else:
        x = region[0]
        y = region[1]
        w = region[2]
        h = region[3]
        cx = x+w/2
        cy = y+h/2
    return cx, cy, w, h
