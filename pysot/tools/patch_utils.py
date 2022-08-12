import os
import cv2
import torch
import json
import random
import numpy as np
from PIL import Image
import sys
sys.path.append('/home/xiangyu/patch-attack/pysot-master')
from pysot.utils.bbox import * 

def MI_Bbox(img=None, height_start=None, height_end=None, width_start=None, width_end=None, magnitude=None, k=None, mu=None):
    patch_img = (np.array(img[height_start:height_end, width_start:width_end])/255)
    original_patch = np.array(patch_img)
    current_gradient = None
    for i in range(int(magnitude*255)):
        for j in range(k):
            new_patch_img = patch_img
            gradients = []
            for c in range(3):
                noise = np.random.normal(0, 1, (height_end-height_start, width_end-width_start))
                gradient = noise / np.linalg.norm(noise, ord=1)
                gradient = np.reshape(gradient, (height_end-height_start, width_end-width_start, 1))
                gradients.append(gradient)
            concat_gradient = np.concatenate([gradients[0], gradients[1], gradients[2]], axis=2)
            if current_gradient is not None:
                new_gradient = mu * current_gradient + concat_gradient
            else:
                new_gradient = concat_gradient
            step_size = magnitude / max((np.abs(np.min(new_gradient)), np.abs(np.max(new_gradient))))
            new_patch_img += (new_gradient * step_size)

        patch_img = new_patch_img
        current_gradient = new_gradient
    patch_img = original_patch + np.clip(patch_img-original_patch, -8/255, 8/255)
    print((255*(patch_img-original_patch)).astype(int))
    img[height_start:height_end, width_start:width_end] = (patch_img*255).astype(int)
    cv2.imwrite('/home/xiangyu/patch-attack/pysot-master/example.jpg', img)    

def get_sorted_overlaps(parser=None):
    overlaps = {}
    target_path = os.path.join(parser.data_base_path, parser.dataset, parser.video, 'predict.txt')
    attack_base_path = os.path.join(parser.data_base_path, parser.dataset, parser.video, 'attack_results', parser.model_name)
    for candidate_name in os.listdir(attack_base_path):
        #Here we select perturbations from all candidate examples.
        txts_dir = os.path.join(attack_base_path, candidate_name, 'attack_txts')
        for txt_file in os.listdir(txts_dir):
            txt_path = os.path.join(txts_dir, txt_file)
            overlaps[candidate_name+'-'+txt_file.split('.')[0]] = overlap_cal(txt_path, target_path)
    overlaps_results = sorted(overlaps.items(), key = lambda x: x[1])
    return overlaps_results

def read_img(parser=None):
    img_path = None
    dataset_json = json.load(open(os.path.join(parser.data_base_path, parser.dataset, parser.dataset+'.json'), 'r'))
    seq_names = []
    for key in dataset_json.keys():
        seq_names.append(key)
    video_name = random.choice(seq_names)
    video_img_path = os.path.join(parser.data_base_path, parser.dataset, video_name, 'img')
    for file in os.listdir(video_img_path):
        if file.split('.')[-1] == 'jpg':
            if file.split('.')[0][-1] == '1':
                img_path = os.path.join(video_img_path, file)
                print(img_path)
                break
    return img_path

            

if __name__ == '__main__':
    '''
    base_noise = np.zeros((30, 20, 3))
    for i in range(8):
        gradients = []
        for c in range(3):
            noise = np.random.normal(0, 1, (30, 20))
            gradient = noise / np.linalg.norm(noise, ord=1)
            gradient = np.reshape(gradient, (30, 20, 1))
            gradients.append(gradient)
    
        concat_gradient = np.concatenate([gradients[0], gradients[1], gradients[2]], axis=2)
        step_size = (8/255) / max((np.abs(np.min(concat_gradient)), np.abs(np.max(concat_gradient))))
        noise = ((concat_gradient * step_size)*255).astype(int)
        base_noise += noise
        base_noise = np.clip(base_noise, -8, 8)
    print(base_noise)
    '''

    patch_example = np.ones((100, 100, 3))
    direct_select(patch_example)
    
    