# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import argparse
import os
import zipfile
import numpy as np
import h5py
import json
from glob import glob
from shutil import rmtree
import pandas as pd

import sys
sys.path.append('./')
from common.parkour_dataset import ParkourDataset
from common.camera import world_to_camera, project_to_2d, image_coordinates
from common.utils import wrap

target_dir = './data'
output_filename        = os.path.join(target_dir, 'data_3d_parkour')
output_filename_2d_det = os.path.join(target_dir, 'data_2d_parkour_pt_coco')
output_filename_2d     = os.path.join(target_dir, 'data_2d_parkour_gt')
subjects = ['PKFC', 'PKLP', 'PKLT', 'PKMT', 'PKRY'] #PKRY is not used

#Movements
#Safety-vault (sv), kong vault (kv), pull-up (pu), muscle-up (mu)
filter_movements = ['sv', 'kv', 'pu', 'mu']

#Assume mass of subject
subject_mass = 74.6

coco_metadata = {
    'layout_name': 'coco',
    'num_joints': 17,
    'keypoints_symmetry': [
        [1, 3, 5, 7, 9, 11, 13, 15],
        [2, 4, 6, 8, 10, 12, 14, 16],
    ]
}

h36m_metadata = {
    'layout_name': 'h36m',
    'num_joints': 17,
    'keypoints_symmetry': [
        [4, 5, 6, 11, 12, 13],
        [1, 2, 3, 14, 15, 16],
    ]
}

parkour_metadata = {
    'layout_name': 'parkour',
    'num_joints': 16,
    'keypoints_symmetry': [
        [0, 5, 6, 7, 8, 13, 14, 15],
        [1, 2, 3, 4, 9, 10, 11, 12],
    ]
}

if __name__ == '__main__':
    if os.path.basename(os.getcwd()) != 'data':
        print('This script must be launched from the "data" directory')
        exit(0)
        
    if os.path.exists(output_filename + '.npz'):
        print('Overwritting', output_filename + '.npz')
        
    print('Converting original Parkour dataset')
    output = {}
    output_2d_det = {}
    
    motion = pd.read_pickle(r'data/Parkour-dataset/gt_motion_forces/joint_3d_positions.pkl')
    forces = pd.read_pickle(r'data/Parkour-dataset/gt_motion_forces/contact_forces_local.pkl') #6-DOF: Linear forces (3) + Moments (3)
    json_source = 'data/Parkour-dataset/outputs_json'

    for key in motion.keys():
        mvmt = key[:4]
        subj = key[-4:]

        keep_sample = False
        for fm in filter_movements:
            if fm in mvmt:
                keep_sample = True

        if not keep_sample:
            continue

        if subj not in output:
            output[subj] = {}
            output_2d_det[subj] = {}

        with open(os.path.join(json_source, key+'.json'), 'r') as f:
            dets_2d = json.load(f)

        poses_2d = []
        for frm in sorted(dets_2d.keys()):
            poses_2d.append(dets_2d[frm]['keypoints'])
        poses_2d = np.array(poses_2d).reshape(-1,17,3)

        positions = motion[key]['joint_3d_positions'] #Expected shape: (T x 16 x 3)
        #Add dummy keypoint, so there's a total of 17. Makes everything else easier
        positions = np.concatenate((positions, positions[:,None,-1]), axis=1)
        fps   = motion[key]['fps']

        l_ankle = forces[key]['contact_forces_local'][:,0]
        r_ankle = forces[key]['contact_forces_local'][:,1]
        l_fingers = forces[key]['contact_forces_local'][:,2]
        r_fingers = forces[key]['contact_forces_local'][:,3]

        #Use only lower body forces
        grfs  = np.concatenate((l_ankle[:,:3], r_ankle[:,:3]), axis=1)

        #Normalize forces by subject mass
        mass = subject_mass
        grfs /= mass

        #All 2D detections are somehow exactly 3 frames behind mocap. So trim mocap
        grfs = grfs[:len(poses_2d)]
        positions = positions[:len(poses_2d)]

        #positions /= 1000 # Supposed to originally be in meters, but scale seems to large. Reduce scale by 1000
        output[subj][mvmt] = {'positions':positions.astype('float32'), 'forces':grfs, 'seq_name':key}
        output_2d_det[subj][mvmt] = [poses_2d]
    
    print('Saving '+output_filename)
    np.savez_compressed(output_filename, data=output)
    
    print('Done.')
    
    #Create 2D detections pose file
    print('')
    print('Saving detection 2D poses...')

    print('Saving '+output_filename_2d_det)
    np.savez_compressed(output_filename_2d_det, positions_2d=output_2d_det, metadata=coco_metadata)
    
    print('Done.')

    # Create 2D pose file
    print('')
    print('Computing ground-truth 2D poses...')
    dataset = ParkourDataset(output_filename + '.npz')
    output_2d_poses = {}
    for subject in dataset.subjects():
        output_2d_poses[subject] = {}
        for action in dataset[subject].keys():
            anim = dataset[subject][action]
            
            positions_2d = []
            for cam in anim['cameras']:
                pos_3d = world_to_camera(anim['positions'], R=cam['orientation'], t=cam['translation'])
                pos_2d = wrap(project_to_2d, pos_3d, cam['intrinsic'], unsqueeze=True)
                pos_2d_pixel_space = image_coordinates(pos_2d, w=cam['res_w'], h=cam['res_h'])
                positions_2d.append(pos_2d_pixel_space.astype('float32'))
            output_2d_poses[subject][action] = positions_2d
            
    print('Saving...')
    metadata = {
        'num_joints': dataset.skeleton().num_joints(),
        'keypoints_symmetry': [dataset.skeleton().joints_left(), dataset.skeleton().joints_right()]
    }
    np.savez_compressed(output_filename_2d, positions_2d=output_2d_poses, metadata=metadata)
    
    print('Done.')
