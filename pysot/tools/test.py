from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import os

import cv2
import torch
import numpy as np
import sys
sys.path.append('/home/xiangyu/patch-attack/pysot-master')
from pysot.core.config import cfg
from pysot.models.model_builder import ModelBuilder
from pysot.tracker.tracker_builder import build_tracker
from pysot.utils.bbox import *
from pysot.utils.model_load import load_pretrain
from toolkit.datasets import DatasetFactory
from toolkit.utils.region import vot_overlap, vot_float2str
import random
from patch_utils import *

parser = argparse.ArgumentParser(description='siamrpn tracking')
parser.add_argument('--dataset', default='OTB', type=str,
        help='datasets')
parser.add_argument('--config', default='/home/xiangyu/patch-attack/pysot-master/experiments/siamrpn_r50_l234_dwxcorr/siamrpn_r50_l234_dwxcorr.yaml', type=str,
        help='config file')
parser.add_argument('--snapshot', default='/home/xiangyu/patch-attack/pysot-master/experiments/siamrpn_r50_l234_dwxcorr/siamrpn_r50_l234_dwxcorr.pth', type=str,
        help='snapshot of models to eval')
parser.add_argument('--model_name', default='siamrpn_r50_l234_dwxcorr', type=str, help='model name')
parser.add_argument('--video', default='Basketball', type=str, help='eval one special video')
parser.add_argument('--video_idx', default=0, type=int, help='check the video index')
parser.add_argument('--patch_num', default=12, type=int, help='specify the num of patches either in height or width')
parser.add_argument('--vis', action='store_true', help='whether visualzie result')
parser.add_argument('--num_video', default=20, type=int, help='number of videos for patch selection')
parser.add_argument('--data_base_path', default='/home/xiangyu/patch-attack/pysot-master/testing_dataset')
parser.add_argument('--attack', default=True, type=bool, help='decide whether to attack or not.')
parser.add_argument('--mu', default=0.5, type=float, help='control the gradient moment.')
parser.add_argument('--k', default=10, type=int, help= 'number of candidate gradient direction in each noise level.')
parser.add_argument('--magnitude', default=8/255, type=float, help='bound for Lp Norm.')
parser.add_argument('--patch_test_num', default=1, type=int, help='Number of patches tested to add momentum-based noise on.')
parser.add_argument('--momentum_over', default=False, type=bool, help='Decide whether to add momentum-based noise over the heavily perturbed frame.')
parser.add_argument('--grain_size', default=1, type=int, help='Decide whether to go fine-grained or coarse-grained in key patch selection module.')
parser.add_argument('--threshold', default=0.2, type=float, help='overlap scaling factor after key patch selection.')
parser.add_argument('--fail_times', default=10, type=float, help='Number of times to tolerate the same pixel-wise mask.')
args = parser.parse_args()

torch.set_num_threads(1)

cfg.merge_from_file(args.config)

cur_dir = os.path.dirname(os.path.realpath(__file__))
dataset_root = os.path.join(args.data_base_path, args.dataset)

# create model
model = ModelBuilder()

# load model
model = load_pretrain(model, args.snapshot).cuda().eval()

# build tracker
tracker = build_tracker(model)

# create dataset
dataset = DatasetFactory.create_dataset(name=args.dataset,
                                    dataset_root=dataset_root,
                                    load_img=False)

model_name = args.snapshot.split('/')[-1].split('.')[0]
total_lost = 0

if args.attack:
    #location info for attack sequences
    patch_h = None
    patch_w = None
    loc_item = None
    loc_index = None
    
    #info for cropped patch and its name
    cropped_example=None
    candidate_name=None
    
    #position of cropped patches
    height_start = 0
    height_end = 0
    width_start = 0
    width_end = 0
else:
    pass

def draw_rect(video=None):
    for v_idx, video in enumerate(dataset):
        if args.video != '':
            if video.name != args.video:
                continue
        for idx, (img, gt_bbox) in enumerate(video):
            if idx == 0:
                gt_bbox = list(map(int, gt_bbox))
                cv2.rectangle(img, (gt_bbox[0], gt_bbox[1]),(gt_bbox[0]+gt_bbox[2], gt_bbox[1]+gt_bbox[3]), (0, 255, 0), 2)
                cv2.imwrite(os.path.join(args.data_base_path,'example.jpg'), img)
            else:
                pass

def get_initial_img():
    initial_img = None
    initial_gt_bbox = None
    for v_idx, video in enumerate(dataset):
        if args.video != '':
            if video.name != args.video:
                continue
        for idx, (img, gt_bbox) in enumerate(video):
            if idx == 0:
                initial_img = img
                initial_gt_bbox = gt_bbox
            else:
                pass
    return initial_img, initial_gt_bbox

def get_video_form():
    for v_idx, video in enumerate(dataset):
        if args.video != '':
            if video.name != args.video:
                continue
        print('Now we are processing video '+ video.name)
        for idx, (img, gt_bbox) in enumerate(video):
            if idx == 0:
                init_gt_bbox = gt_bbox
                img_height = img.shape[0]
                img_width = img.shape[1]
                index_list = index_cal(img_height, img_width, gt_bbox, args)
            else:
                pass
    return init_gt_bbox, index_list

def draw(candidate_name, height_start, height_end, width_start, width_end, attack_idx, attack_loc):
    attack_path = os.path.join(args.data_base_path, args.dataset, args.video, 'attack_results', args.model_name, candidate_name,
                              'attack_txts', '{}-{}-{}-{}-{}-{}-{}.txt'.format(attack_idx, height_start, height_end, 
                              width_start, width_end, attack_loc[0], attack_loc[1]))
    predict_path = os.path.join(args.data_base_path, args.dataset, args.video, 'predict.txt')
    attack_lines = open(attack_path, 'r').readlines()
    predict_lines = open(predict_path, 'r').readlines()
    img_save_path = os.path.join(args.data_base_path, args.dataset, args.video, 'attack_results', args.model_name, candidate_name, 
                              'attack_imgs')
    if not os.path.isdir(img_save_path):
        os.makedirs(img_save_path)
    for v_idx, video in enumerate(dataset):
        if video.name != args.video:
            continue
        else:
            pass
        for idx, (img, gt_bbox) in enumerate(video):
            if idx == 0:
                continue
            else:
                pass
            gt_bbox = list(map(int, gt_bbox))
            attack_bbox = attack_lines[idx][:-1].split(',')
            predict_bbox = predict_lines[idx][:-1].split(',')
            for s in range(len(attack_bbox)):
                attack_bbox[s] = float(attack_bbox[s])
            for s in range(len(predict_bbox)):
                predict_bbox[s] = float(predict_bbox[s])
            attack_bbox = list(map(int, attack_bbox))
            predict_bbox = list(map(int, predict_bbox))
            cv2.rectangle(img, (gt_bbox[0], gt_bbox[1]),(gt_bbox[0]+gt_bbox[2], gt_bbox[1]+gt_bbox[3]), (0, 255, 0), 2)
            cv2.rectangle(img, (attack_bbox[0], attack_bbox[1]), (attack_bbox[0]+attack_bbox[2], attack_bbox[1]+attack_bbox[3]), (0, 0, 255), 2)
            cv2.rectangle(img, (predict_bbox[0], predict_bbox[1]), (predict_bbox[0]+predict_bbox[2], predict_bbox[1]+predict_bbox[3]), (255, 0, 0), 2)

            cv2.imwrite(os.path.join(img_save_path, '{}-{}-{}-{}-{}-{}-{}-{}.jpg'.format(attack_idx, height_start, height_end, width_start,
            width_end, attack_loc[0], attack_loc[1], idx)), img)

def pixel_processed_img(pixel_mask=None, img=None, cropped_img=None, noise=None, loc_y=None, loc_x=None):
    
    height = cropped_img.shape[0]
    width = cropped_img.shape[1]
    base_img = np.array(img)
    masked_ori_img = np.array(img[loc_y:loc_y+height, loc_x:loc_x+width, :])
    reverse_mask = np.array(pixel_mask)
    patch = None

    for i in range(len(reverse_mask)):
        for j in range(len(reverse_mask[i])):
            for k in range(len(reverse_mask[i][j])):
                if reverse_mask[i][j][k] == 1:
                    reverse_mask[i][j][k] = 0
                else:
                    reverse_mask[i][j][k] = 1
    
    masked_ori_img *= reverse_mask
    
    if noise is not None:
        if not args.momentum_over:
            base_img += noise
            base_img[loc_y:loc_y+height, loc_x:loc_x+width, :] = cropped_img
        else:
            base_img[loc_y:loc_y+height, loc_x:loc_x+width, :] = cropped_img
            base_img += noise
    else:
        base_img[loc_y:loc_y+height, loc_x:loc_x+width, :] = cropped_img

    unmasked_ori_img = np.array(base_img[loc_y:loc_y+height, loc_x:loc_x+width, :])
    unmasked_ori_img *= pixel_mask
    
    patch = masked_ori_img + unmasked_ori_img

    base_img[loc_y:loc_y+height, loc_x:loc_x+width, :] = patch

    return base_img

def test_direct(initial_img=None):
    pred_bboxes = []
    for v_idx, video in enumerate(dataset):
        if video.name == args.video:
            pass
        else:
            continue

        for idx, (img, gt_bbox) in enumerate(video):
            if idx == 0:
                cx, cy, w, h = get_axis_aligned_bbox(np.array(gt_bbox))
                gt_bbox_ = [cx-(w-1)/2, cy-(h-1)/2, w, h]
                tracker.init(initial_img, gt_bbox_)
                pred_bbox = gt_bbox_
                pred_bboxes.append(pred_bbox)
            else:
                outputs = tracker.track(img)
                pred_bbox = outputs['bbox']                        
                pred_bboxes.append(pred_bbox)

    with open(os.path.join(args.data_base_path, 'example-'+args.video+'.txt'), 'w') as f:
        for x in pred_bboxes:
            f.write(','.join([str(i) for i in x])+'\n')
    f.close()
    
    source_path = os.path.join(args.data_base_path, 'example-'+args.video+'.txt')
    target_path = os.path.join(args.data_base_path, args.dataset, args.video, 'predict.txt')
    overlap = overlap_cal(source_path, target_path)

    os.remove(source_path)
    
    return overlap

def direct_select(grain_size=None, initial_overlap=None, candidate_name=None, img=None, cropped_img=None, noise=None, loc_y=None, loc_x=None):

    q_times = 0
    initial_width = cropped_img.shape[1]
    initial_height = cropped_img.shape[0]
    pixel_mask = None
    pixel_masks = []
    previous_mask = None
    rd = 0
    fail_list = []

    patch_select_imgs_path = os.path.join(args.data_base_path, args.dataset, args.video, 'attack_results', 
                                           args.model_name, candidate_name, 'patch_select_imgs')
    if not os.path.isdir(patch_select_imgs_path):
        os.makedirs(patch_select_imgs_path)
    
    while True:
        
        rd += 1
        print('This is round {}\'s search. '.format(rd))
        if pixel_mask is None:
            pixel_mask = np.ones((initial_height, initial_width, 3))
            previous_mask = np.array(pixel_mask)

        else:
            pass

        scaled_shape = [0, 0, initial_height, initial_width]
        fail = 1
        
        while scaled_shape[2] > grain_size or scaled_shape[3] > grain_size :
            
            choice = None

            if scaled_shape[2] > grain_size and scaled_shape[3] == grain_size:
                random_direct = 0
            elif scaled_shape[3] > grain_size and scaled_shape[2] == grain_size:
                random_direct = 1
            else:
                random_direct = np.random.randint(0, 2)

            #Cut the initial frame randomly from two axis
            if random_direct == 0:

                mid_height = scaled_shape[2] // 2 
                #Test y-down directions
                pixel_mask_copy_down = np.ones((initial_height, initial_width, 3))
                pixel_mask_copy_down[scaled_shape[0] + mid_height : scaled_shape[0] + scaled_shape[2], scaled_shape[1] : scaled_shape[1] + scaled_shape[3], :] = 0
                processed_mask = (pixel_mask * pixel_mask_copy_down).astype('uint8')
                processed_img = pixel_processed_img(pixel_mask=processed_mask, img=img, cropped_img=cropped_img, 
                                                noise=noise, loc_y=loc_y, loc_x=loc_x)
                overlap_score_down = test_direct(initial_img = processed_img)
                q_times += 1
                print('overlap_score_down is {}, round {}, query times {}'.format(overlap_score_down, rd, q_times))

                #Test y-up directions
                pixel_mask_copy_up = np.ones((initial_height, initial_width, 3))
                pixel_mask_copy_up[scaled_shape[0] : scaled_shape[0] + mid_height, scaled_shape[1] : scaled_shape[1] + scaled_shape[3], :] = 0
                processed_mask = (pixel_mask * pixel_mask_copy_up).astype('uint8')
                processed_img = pixel_processed_img(pixel_mask=processed_mask, img=img, cropped_img=cropped_img, 
                                                noise=noise, loc_y=loc_y, loc_x=loc_x)
                overlap_score_up = test_direct(initial_img = processed_img)
                q_times += 1
                print('overlap_score_up is {}, round {}, query times {}'.format(overlap_score_up, rd, q_times))

                #lower score means better patch selection results
                sig_down = 1 / (1 + np.exp((overlap_score_down / (overlap_score_down + overlap_score_up)) - 1))
                sig_up = 1 / (1 + np.exp(-overlap_score_down / (overlap_score_down + overlap_score_up)))
                prob_down = sig_down / (sig_down + sig_up)

                if initial_overlap <= 0.1:
                    if overlap_score_down < overlap_score_up:
                        if overlap_score_down <= 3 * initial_overlap:
                            choice = 1
                        else:
                            choice = np.random.binomial(1, prob_down, 1)[0]
                    elif overlap_score_down >= overlap_score_up:
                        if overlap_score_up <= 3 * initial_overlap:
                            choice = 0
                        else:
                            choice = np.random.binomial(1, prob_down, 1)[0]
                elif initial_overlap > 0.1 and initial_overlap <= 0.4:
                    if overlap_score_down < overlap_score_up:
                        if overlap_score_down <= 1.5 * initial_overlap:
                            choice = 1
                        else:
                            choice = np.random.binomial(1, prob_down, 1)[0]
                    elif overlap_score_down >= overlap_score_up:
                        if overlap_score_up <= 1.5 * initial_overlap:
                            choice = 0
                        else:
                            choice = np.random.binomial(1, prob_down, 1)[0]
                else:
                    if overlap_score_down < overlap_score_up:
                        if overlap_score_down < 0.4:
                            choice = 1
                        else:
                            choice = np.random.binomial(1, prob_down, 1)[0]
                    if overlap_score_down >= overlap_score_up:
                        if overlap_score_up < 0.4:
                            choice = 0
                        else:
                            choice = np.random.binomial(1, prob_down, 1)[0]

            elif random_direct == 1:

                mid_width = scaled_shape[3] // 2
                
                #Test x-right directions
                pixel_mask_copy_right = np.ones((initial_height, initial_width, 3))
                pixel_mask_copy_right[scaled_shape[0] : scaled_shape[0] + scaled_shape[2], scaled_shape[1] + mid_width: scaled_shape[1] + scaled_shape[3], :] = 0
                processed_mask = (pixel_mask_copy_right * pixel_mask).astype('uint8')
                processed_img = pixel_processed_img(pixel_mask=processed_mask, img=img, cropped_img=cropped_img, 
                                                noise=noise, loc_y=loc_y, loc_x=loc_x)
                overlap_score_right = test_direct(initial_img = processed_img)
                q_times += 1
                print('overlap_score_right is {}, round {}, query times {}'.format(overlap_score_right, rd, q_times))

                #Test x-left directions
                pixel_mask_copy_left = np.ones((initial_height, initial_width, 3))
                pixel_mask_copy_left[scaled_shape[0] : scaled_shape[0] + scaled_shape[2] , scaled_shape[1] : scaled_shape[1] + mid_width, :] = 0
                processed_mask = (pixel_mask_copy_left * pixel_mask).astype('uint8')
                processed_img = pixel_processed_img(pixel_mask=processed_mask, img=img, cropped_img=cropped_img, 
                                                noise=noise, loc_y=loc_y, loc_x=loc_x)
                overlap_score_left = test_direct(initial_img = processed_img)
                q_times += 1
                print('overlap_score_left is {}, round {}, query times {}'.format(overlap_score_left, rd, q_times))

                #lower score means better patch selection results
                sig_right = 1 / (1 + np.exp((overlap_score_right / (overlap_score_right + overlap_score_left)) - 1))
                sig_left = 1 / (1 + np.exp(-overlap_score_right / (overlap_score_right + overlap_score_left)))
                prob_right = sig_right / (sig_right + sig_left)
                
                if initial_overlap <= 0.1:
                    if overlap_score_right < overlap_score_left:
                        if overlap_score_right <= 3 * initial_overlap:
                            choice = 1
                        else:
                            choice = np.random.binomial(1, prob_right, 1)[0]
                    elif overlap_score_right >= overlap_score_left:
                        if overlap_score_left <= 3 * initial_overlap:
                            choice = 0
                        else:
                            choice = np.random.binomial(1, prob_right, 1)[0]
                elif initial_overlap > 0.1 and initial_overlap <= 0.4:
                    if overlap_score_right < overlap_score_left:
                        if overlap_score_right <= 1.5 * initial_overlap:
                            choice = 1
                        else:
                            choice = np.random.binomial(1, prob_right, 1)[0]
                    elif overlap_score_right >= overlap_score_left:
                        if overlap_score_left <= 1.5 * initial_overlap:
                            choice = 0
                        else:
                            choice = np.random.binomial(1, prob_right, 1)[0]
                else:
                    if overlap_score_right < overlap_score_left:
                        if overlap_score_right < 0.4:
                            choice = 1
                        else:
                            choice = np.random.binomial(1, prob_right, 1)[0]
                    if overlap_score_right >= overlap_score_left:
                        if overlap_score_left < 0.4:
                            choice = 0
                        else:
                            choice = np.random.binomial(1, prob_right, 1)[0]
                       
            dot_matrix = np.ones((initial_height, initial_width, 3))
            if random_direct == 0:
                if choice == 1:
                    dot_matrix[scaled_shape[0]+(scaled_shape[2]//2):scaled_shape[0]+scaled_shape[2], scaled_shape[1]:scaled_shape[1]+scaled_shape[3], :] = 0
                    scaled_shape[0] += (scaled_shape[2]//2)
                else:
                    dot_matrix[scaled_shape[0]:scaled_shape[0]+(scaled_shape[2]//2), scaled_shape[1]:scaled_shape[1]+scaled_shape[3], :] = 0
                scaled_shape[2] //= 2
            elif random_direct == 1:
                if choice == 1:
                    dot_matrix[scaled_shape[0]:scaled_shape[0]+scaled_shape[2], scaled_shape[1]+(scaled_shape[3]//2):scaled_shape[1]+scaled_shape[3], :] = 0
                    scaled_shape[0] += (scaled_shape[2]//2)    
                else:
                    dot_matrix[scaled_shape[0]:scaled_shape[0]+scaled_shape[2], scaled_shape[1]:scaled_shape[1]+(scaled_shape[3]//2), :] = 0
                scaled_shape[3] //= 2
            pixel_mask_inter = pixel_mask * dot_matrix
            
            print('Random Direct is {}, choice is {}'.format(random_direct, choice))
            print('The scaled shape is ({}, {})'.format(scaled_shape[2], scaled_shape[3]))
            
            if (pixel_mask_inter == previous_mask).all():
                print('The Newly Generated mask is identical to previously generated one.')
                break
            else:
                processed_img = pixel_processed_img(pixel_mask=pixel_mask_inter.astype('uint8'), img=img, cropped_img=cropped_img, 
                                                noise=noise, loc_y=loc_y, loc_x=loc_x)
                overlap_score_milestone = test_direct(initial_img = processed_img)
                q_times += 1
                print('We got the new overlap score to be {}. This is the round {} with query times {}'.format(overlap_score_milestone, rd, q_times))

                if overlap_score_milestone < initial_overlap:
                    pixel_mask = np.array(pixel_mask_inter)
                    previous_mask = np.array(pixel_mask)
                    fail = 0
                    print('initial overlap is {}, overlap_score_milestone is {}, condition satisfied.'.format(initial_overlap, overlap_score_milestone))
                    cv2.imwrite(os.path.join(patch_select_imgs_path, str(loc_y)+'-'+str(loc_x)+'-'+str(rd)+'-'+str(q_times)+'.jpg'), processed_img)
                    break
                
                else:
                    flag = False
                    if initial_overlap <= 0.1:
                        if overlap_score_milestone <= 0.3:
                            print('initial overlap is {}, overlap_score_milestone is {}, condition satisfied.'.format(initial_overlap, overlap_score_milestone))
                            flag = True
                    elif initial_overlap > 0.1 and initial_overlap <= 0.4:
                        if overlap_score_milestone <= 0.6:
                            print('initial overlap is {}, overlap_score_milestone is {}, condition satisfied.'.format(initial_overlap, overlap_score_milestone))
                            flag = True
                    else:
                        if (overlap_score_milestone <= 1.2 * initial_overlap) and overlap_score_milestone < 1.0:
                            print('initial overlap is {}, overlap_score_milestone is {}, condition satisfied.'.format(initial_overlap, overlap_score_milestone))
                            flag = True
                    if flag:
                        pixel_mask = np.array(pixel_mask_inter)
                        previous_mask = np.array(pixel_mask)
                        fail = 0
                        cv2.imwrite(os.path.join(patch_select_imgs_path, str(loc_y)+'-'+str(loc_x)+'-'+str(rd)+'-'+str(q_times)+'.jpg'), processed_img)
                        break
                    else:
                        if scaled_shape[2] == grain_size and scaled_shape[3] == grain_size:
                            pixel_mask = np.array(previous_mask)

        fail_list.append(fail)
        fail_count = 0
        index = -1
        while np.abs(index) <= len(fail_list):
            if fail_list[index] == 1:
                fail_count += 1
            else:
                break
            index -= 1

        if fail_count >= args.fail_times:
            break
            
    print('We processed the mask with {} round, totally {} query times.'.format(rd, q_times))

    return pixel_mask, q_times

def momentum_add(overlap_score=None, cropped_img=None, candidate_name=None, loc_y=None, loc_x=None, magnitude=None, k=None, mu=None):
    
    #Get the initial frame, copy it to the template
    initial_img, init_gt_bbox = get_initial_img()
    current_gradient = None
    current_noise = None

    best_overlap_score = overlap_score
    best_attack_score = 1.0
    best_level = 0
    best_iter = 0

    level_gradient = None
    level_overlap_score = 1.0

    
    if not args.momentum_over:
        momentum_attack_txt_dir = os.path.join(args.data_base_path, args.dataset, args.video, 'attack_results', args.model_name, candidate_name, 'under_momentum_attack_txts')
        momentum_attack_img_dir = os.path.join(args.data_base_path, args.dataset, args.video, 'attack_results', args.model_name, candidate_name, 'under_momentum_attack_imgs')
    else:
        momentum_attack_txt_dir = os.path.join(args.data_base_path, args.dataset, args.video, 'attack_results', args.model_name, candidate_name, 'over_momentum_attack_txts')
        momentum_attack_img_dir = os.path.join(args.data_base_path, args.dataset, args.video, 'attack_results', args.model_name, candidate_name, 'over_momentum_attack_imgs')
    if not os.path.isdir(momentum_attack_txt_dir):
        os.makedirs(momentum_attack_txt_dir)
    if not os.path.isdir(momentum_attack_img_dir):
        os.makedirs(momentum_attack_img_dir)

    for i in range(int(magnitude*255)):
        better_flag = False
        new_overlap_scores = {}

        for j in range(k):
            new_initial_img = np.array(initial_img)
            gradients = []
            for c in range(3):
                noise = np.random.normal(0, 1, (initial_img.shape[0], initial_img.shape[1]))
                gradient = noise / np.linalg.norm(noise, ord=1)
                gradient = np.reshape(gradient, (initial_img.shape[0], initial_img.shape[1], 1))
                gradients.append(gradient)
            concat_gradient = np.concatenate([gradients[0], gradients[1], gradients[2]], axis=2)
            if current_gradient is not None:
                new_gradient = mu * np.array(current_gradient) + concat_gradient
            else:
                new_gradient = concat_gradient
            
            #Now time current updated gradient with step size from 1/255 to 8/255
            if i == 0:
                new_noise = (np.sign(new_gradient)).astype(np.uint8)
            else:
                new_noise = current_noise + (np.sign(new_gradient)).astype(np.uint8)
            
            if not args.momentum_over:
                new_initial_img += new_noise
                new_initial_img[loc_y:loc_y+cropped_img.shape[0], loc_x:loc_x+cropped_img.shape[1]] = cropped_img
            else:
                new_initial_img[loc_y:loc_y+cropped_img.shape[0], loc_x:loc_x+cropped_img.shape[1]] = cropped_img
                new_initial_img += new_noise
            
            momentum_txt_path = os.path.join(momentum_attack_txt_dir, str(loc_y)+'-'+str(loc_x) + '-' + str(i+1) + '-' + str(j)+'.txt')
            
            print('Test momentum-based perturbation adding algorithm at location {}-{} in the level {}, iteration {}'.format(loc_y, loc_x, i+1, j))
            new_overlap_score = test_momentum(initial_frame=new_initial_img, candidate_name=candidate_name, 
                                        height_start=loc_y, width_start=loc_x, momentum_txt_path=momentum_txt_path)
            print('The new_ovelap_score is {}'.format(new_overlap_score))
            print('---------------------------')
            new_overlap_scores[j] = (new_overlap_score, new_gradient, new_noise)
        

        for idx in new_overlap_scores.keys():
            if new_overlap_scores[idx][0] < level_overlap_score:
                level_overlap_score = new_overlap_scores[idx][0]
                level_gradient = new_overlap_scores[idx][1]
                level_noise = new_overlap_scores[idx][2]
            else:
                pass
            if new_overlap_scores[idx][0] < best_overlap_score:

                better_flag = True
                best_overlap_score = new_overlap_scores[idx][0]
                current_gradient = new_overlap_scores[idx][1]
                current_noise = new_overlap_scores[idx][2]

                best_level = i+1
                best_iter = idx
                base_img = np.array(initial_img)
                
                if not args.momentum_over:
                    base_img += current_noise
                    base_img[loc_y:loc_y+cropped_img.shape[0], loc_x:loc_x+cropped_img.shape[1]] = cropped_img
                else:
                    base_img[loc_y:loc_y+cropped_img.shape[0], loc_x:loc_x+cropped_img.shape[1]] = cropped_img
                    base_img += current_noise

                cv2.imwrite(os.path.join(momentum_attack_img_dir, str(loc_y) + '-' + str(loc_x) + '-' + str(i+1) + '-' + str(idx) + '.jpg'), base_img)
            else:
                pass
        
        if better_flag:
            pass
        else:
            current_gradient = level_gradient
            current_noise = level_noise
        
        print('-----------------------------------------------------------------------------------------')
        print('The best overlap score after {}-th iteration at level {} is {}'.format(k, i+1, level_overlap_score))
    
    print('---------------------------------------------------------------------------------------')
    print('The best overlap score overall is {} at level {} from the {}-th iteration.'.format(best_overlap_score, best_level, best_iter))
    
    if best_level == 0 and best_iter == 0:
        current_noise = None

    return initial_img, current_noise, best_overlap_score

def test_momentum(initial_frame=None, candidate_name=None, height_start=None, width_start=None, momentum_txt_path=None):
    pred_bboxes = []
    scores = []
    for v_idx, video in enumerate(dataset):
        if video.name == args.video:
            pass
        else:
            continue
        for idx, (img, gt_bbox) in enumerate(video):
            if idx == 0:
                cx, cy, w, h = get_axis_aligned_bbox(np.array(gt_bbox))
                img = initial_frame
                gt_bbox_ = [cx-(w-1)/2, cy-(h-1)/2, w, h]
                tracker.init(img, gt_bbox_)
                pred_bbox = gt_bbox_
                pred_bboxes.append(pred_bbox)
            elif idx > 0:
                outputs = tracker.track(img)
                pred_bbox = outputs['bbox']                        
                pred_bboxes.append(pred_bbox)
                scores.append(outputs['best_score'])
        
        with open(momentum_txt_path, 'w') as f:
            for x in pred_bboxes:
                f.write(','.join([str(i) for i in x])+'\n')
        f.close()
    
    target_path = os.path.join(args.data_base_path, args.dataset, args.video, 'predict.txt')
    overlap = overlap_cal(momentum_txt_path, target_path)

    return overlap

def original_predicted(video_name=None):

    pred_bboxes = []
    for v_idx, video in enumerate(dataset):
        if video.name == video_name:
            pass
        else:
            continue
        for idx, (img, gt_bbox) in enumerate(video):
            if idx == 0:
                cx, cy, w, h = get_axis_aligned_bbox(np.array(gt_bbox))
                gt_bbox_ = [cx-(w-1)/2, cy-(h-1)/2, w, h]
                tracker.init(img, gt_bbox_)
                pred_bbox = gt_bbox_
                pred_bboxes.append(pred_bbox)
            else:
                outputs = tracker.track(img)
                pred_bbox = outputs['bbox']                        
                pred_bboxes.append(pred_bbox)
    
    predict_save = os.path.join(args.data_base_path, args.dataset, video_name, 'predict.txt')
    if os.path.exists(predict_save):
        pass
    else:
        with open(predict_save, 'w') as f:
            for x in pred_bboxes:
                f.write(','.join([str(i) for i in x])+'\n')
        f.close()

def test_patch(video_name=None, candidate_name=None, height_start=None, height_end=None, width_start=None, width_end=None, loc_y=None, loc_x=None):

    candidate_video = candidate_name.split('_')[0]
    candidate_frame = candidate_name.split('_')[1]
    candidate_img = cv2.imread(os.path.join(args.data_base_path, args.dataset, candidate_video, 'img', candidate_frame+'.jpg'))
    cropped_img = candidate_img[height_start:height_end, width_start:width_end, :]
    if (height_end - height_start) < patch_h or (width_end - width_start) < patch_w:
        cropped_img = cv2.resize(cropped_img, (patch_w, patch_h), interpolation=cv2.INTER_LINEAR)
    else:
        pass
    cv2.imwrite(os.path.join(args.data_base_path, 'cropped.jpg'), cropped_img)

    pred_bboxes = []
    scores = []
    
    for v_idx, video in enumerate(dataset):
        if video.name == video_name:
            pass
        else:
            continue

        for idx, (img, gt_bbox) in enumerate(video):
            if idx == 0:
                init_gt_bbox = list(map(int, gt_bbox))
                print((init_gt_bbox[3], init_gt_bbox[2]))
                #height start and end, width start and end are unresized values.
                print((height_end-height_start, width_end-width_start))
                img[loc_y:loc_y+init_gt_bbox[3], loc_x:loc_x+init_gt_bbox[2]] = cropped_img
                cv2.imwrite(os.path.join(args.data_base_path, 'example.jpg'), img)
                cx, cy, w, h = get_axis_aligned_bbox(np.array(gt_bbox))
                gt_bbox_ = [cx-(w-1)/2, cy-(h-1)/2, w, h]
                tracker.init(img, gt_bbox_)
                pred_bbox = gt_bbox_
                pred_bboxes.append(pred_bbox)
            else:
                outputs = tracker.track(img)
                pred_bbox = outputs['bbox']                        
                pred_bboxes.append(pred_bbox)
                scores.append(outputs['best_score'])
    
    with open(os.path.join(args.data_base_path, 'example.txt'), 'w') as f:
        for x in pred_bboxes:
            f.write(','.join([str(i) for i in x])+'\n')
    f.close()
    
    source_path = os.path.join(args.data_base_path, 'example.txt')
    target_path = os.path.join(args.data_base_path, args.dataset, video_name, 'predict.txt')
    overlap = overlap_cal(source_path, target_path)

    os.remove(source_path)

    return cropped_img, overlap

    
def test_track(attack=False, candidate_list=None):
    for v_idx, video in enumerate(dataset):
        if args.video != '':
            if video.name != args.video:
                continue
        toc = 0
        scores = []
        track_times = []
            
        if attack:
            save_path = os.path.join(dataset_root, video.name, 'attack_results', args.model_name, candidate_name, 'attack_imgs')
            if not os.path.isdir(save_path):
                os.makedirs(save_path)

            gt_bboxes = []
            pred_bboxes = []
            for idx, (img, gt_bbox) in enumerate(video):
                tic = cv2.getTickCount()
                if idx == 0:
                    print((patch_h, patch_w, 3))
                    print(cropped_example.shape)
                    print((loc_item[0], loc_item[0]+patch_h))
                    print((loc_item[1], loc_item[1]+patch_w))
                    print(img.shape)
                    img[loc_item[0]:loc_item[0]+patch_h, loc_item[1]:loc_item[1]+patch_w] = cropped_example
                    img_save_path = save_path + '/{}-{}-{}-{}-{}-{}-{}-{}.jpg'.format(loc_index, 
                    height_start, height_end, width_start, width_end, loc_item[0], loc_item[1], idx)
                    cv2.imwrite(img_save_path, img)
                    cx, cy, w, h = get_axis_aligned_bbox(np.array(gt_bbox))
                    gt_bbox_ = [cx-(w-1)/2, cy-(h-1)/2, w, h]
                    tracker.init(img, gt_bbox_)
                    pred_bbox = gt_bbox_
                    scores.append(None)
                    if 'VOT2018-LT' == args.dataset:
                        pred_bboxes.append([1])
                    else:
                        pred_bboxes.append(pred_bbox)
                else:
                    outputs = tracker.track(img)
                    pred_bbox = outputs['bbox']                        
                    pred_bboxes.append(pred_bbox)
                    scores.append(outputs['best_score'])
                        
                toc += cv2.getTickCount() - tic
                
            track_times.append((cv2.getTickCount() - tic)/cv2.getTickFrequency())
            toc /= cv2.getTickFrequency()
            print('{} attacked by {} at patch {} within time {}'.format(args.video, candidate_name, loc_index, toc))

            '''save track results'''
            txt_path = os.path.join(dataset_root, video.name, 'attack_results', args.model_name, candidate_name, 'attack_txts')
            if not os.path.isdir(txt_path):
                os.makedirs(txt_path)
            result_path = os.path.join(txt_path, '{}-{}-{}-{}-{}-{}-{}.txt'.format(loc_index, 
            height_start, height_end, width_start, width_end, loc_item[0], loc_item[1]))
            with open(result_path, 'w') as f:
                for x in pred_bboxes:
                    f.write(','.join([str(i) for i in x])+'\n')
        else:
            pred_bboxes = []
            for idx, (img, gt_bbox) in enumerate(video):
                tic = cv2.getTickCount()
                if idx == 0:
                    cx, cy, w, h = get_axis_aligned_bbox(np.array(gt_bbox))
                    gt_bbox_ = [cx-(w-1)/2, cy-(h-1)/2, w, h]
                    tracker.init(img, gt_bbox_)
                    pred_bbox = gt_bbox_
                    scores.append(None)
                    if 'VOT2018-LT' == args.dataset:
                        pred_bboxes.append([1])
                    else:
                        pred_bboxes.append(pred_bbox)
                else:
                    outputs = tracker.track(img)
                    pred_bbox = outputs['bbox']
                    pred_bboxes.append(pred_bbox)
                    scores.append(outputs['best_score'])
                toc += cv2.getTickCount() - tic

            track_times.append((cv2.getTickCount() - tic)/cv2.getTickFrequency())
            toc /= cv2.getTickFrequency()           

        print('({:3d}) Video: {:12s} Time: {:5.1f}s Speed: {:3.1f}fps'.format(v_idx+1, video.name, toc, idx / toc))


if __name__ == '__main__':

    dataset_names = []
    json_content = json.load(open(os.path.join(args.data_base_path, args.dataset, args.dataset+'.json'), 'r+'))
    for key in json_content.keys():
        dataset_names.append(key)
    print(dataset_names)
    
    avg_overlap = 1.0
    gt_bbox, index_list = get_video_form()
    patch_h = gt_bbox[3]
    patch_w = gt_bbox[2]
    height_start, height_end, width_start, width_end, candidate_name, cropped_example = get_cropped_example(parser=args, gt_bbox=gt_bbox)

    original_predicted(video_name=args.video)

    for i, loc in enumerate(index_list):
        loc_index = i
        loc_item = loc
        test_track(attack=True)
    
    draw_idx = 0
    draw_loc = None
    draw_height_start = 0
    draw_height_end = 0
    draw_width_start = 0
    draw_width_end = 0
    for i in range(args.patch_num*2):
        source_path = os.path.join(args.data_base_path, args.dataset, args.video, 'attack_results', args.model_name, 
                                candidate_name, 'attack_txts')
        target_path = os.path.join(args.data_base_path, args.dataset, args.video, 'predict.txt')
        for file in os.listdir(source_path):
            splits = file.split('-')
            if i == int(splits[0]):
                source_path = os.path.join(source_path, file)
                location = (splits[-2], splits[-1][:-4])
                height_start = splits[1]
                height_end = splits[2]
                width_start = splits[3]
                width_end = splits[4]
                new_overlap = overlap_cal(source_path, target_path)
                if new_overlap < avg_overlap:
                    avg_overlap = new_overlap
                    draw_idx = i
                    draw_loc = location
                    draw_height_start = height_start
                    draw_height_end = height_end
                    draw_width_start = width_start
                    draw_width_end = width_end
    draw(candidate_name, draw_height_start, draw_height_end, draw_width_start, draw_width_end, draw_idx, draw_loc)

    
    sorted_overlaps = get_sorted_overlaps(parser=args)
    print(sorted_overlaps)
    for i in range(args.patch_test_num):
        candidate_name = sorted_overlaps[i][0].split('-')[0]
        height_start = int(sorted_overlaps[i][0].split('-')[-6])
        height_end = int(sorted_overlaps[i][0].split('-')[-5])
        width_start = int(sorted_overlaps[i][0].split('-')[-4])
        width_end = int(sorted_overlaps[i][0].split('-')[-3])
        loc_y = int(sorted_overlaps[i][0].split('-')[-2])
        loc_x = int(sorted_overlaps[i][0].split('-')[-1])
        
        if len(sorted_overlaps[i][0].split('-')) > 8:
            for j in range(len(sorted_overlaps[i][0].split('-'))-8):
                candidate_name += ('-' + sorted_overlaps[i][0].split('-')[j+1])
        
        cropped_img, overlap = test_patch(video_name=args.video, candidate_name=candidate_name, height_start=height_start,
                   height_end=height_end, width_start=width_start, width_end=width_end, loc_y=loc_y, loc_x=loc_x)
        print('The ranking-{} heavy patch noise added results is {}'.format(i+1, overlap))
    
        initial_img, current_noise, overlap_momentum = momentum_add(overlap_score=overlap, cropped_img=cropped_img, candidate_name=candidate_name, 
                                                    loc_y=loc_y, loc_x=loc_x, magnitude=args.magnitude, k=args.k, mu=args.mu)
        pixel_mask, q_times = direct_select(grain_size=args.grain_size, initial_overlap=min(overlap_momentum, overlap), candidate_name=candidate_name, 
                                                    img=initial_img, cropped_img=cropped_img, noise=current_noise, loc_y=loc_y, loc_x=loc_x)
                

    


    
