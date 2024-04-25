# Copyright (c) 2018-present, Facebook, Inc.
import numpy as np
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

import sys
sys.path.append('./')
from common.force_pose_dataset import ForcePoseDataset
from common.camera import world_to_camera, project_to_2d, image_coordinates
from common.utils import wrap

target_dir = './data'
output_filename        = os.path.join(target_dir, 'data_3d_force_pose')
output_filename_2d_det = os.path.join(target_dir, 'data_2d_force_pose_pt_coco')
output_filename_2d     = os.path.join(target_dir, 'data_2d_force_pose_gt')
subjects = ['Subject1', 'Subject2', 'Subject3', 'Subject4', 'Subject5', 'Subject6', 'Subject7', 'Subject8']

mocap_markers = ['CLAV', 'LACRM', 'LASIS', 'LHEEL', 'LLAK',\
               'LLEB', 'LLFARM', 'LLHND', 'LLKN', 'LLTOE',\
               'LLWR', 'LMAK', 'LMEB', 'LMFARM', 'LMHND',\
               'LMKN', 'LMTOE', 'LMWR', 'LPSI', 'LSHANK',\
               'LTHIGH', 'LUPARM', 'RACRM', 'RASIS',\
               'RHEEL', 'RLAK', 'RLEB', 'RLFARM', 'RLHND',\
               'RLKN', 'RLTOE', 'RLWR', 'RMAK', 'RMEB', 'RMFARM',\
               'RMHND', 'RMKN', 'RMTOE', 'RMWR', 'RPSI',\
               'RSHANK', 'RTHIGH', 'RUPARM', 'STRM',\
               'T1', 'T10', 'THEAD']

#Use only these movements for training and eval
filter_movements = ['Counter_Movement_Jump',
		    'Single_Leg_Squat',
		    'Single_Leg_Jump',
		    'Squat_Jump',
		    'Squat']

subject_masses = {
		  'Subject1':83.25,
		  'Subject2':86.48,
		  'Subject3':87.54,
		  'Subject4':86.11,
		  'Subject5':74.91,
		  'Subject6':111.91,
                  'Subject7':82.64,
		  'Subject8':90.44}

cameras =[ 
          'cam_17364068',
          'cam_17400877',
          'cam_17400878',
          'cam_17400879',
          'cam_17400880',
          'cam_17400881',
          'cam_17400883',
          'cam_17400884'
         ] 

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

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Force Pose dataset converter')
    parser.add_argument('--source-json', default='./data/force_pose', type=str, metavar='PATH', help='convert original dataset')
    args = parser.parse_args()
    
    if os.path.exists(output_filename + '.npz'):
        print('Overwritting', output_filename + '.npz')
        
    print('Converting original Force Pose dataset from', args.source_json, '(JSON files)')
    output = {}
    output_2d_det = {}
    
    file_list = glob(os.path.join(args.source_json, '*.json'))
    print(file_list)
    #file_list.remove(os.path.join(args.source_json, 'test.json')) #just a symbolic link
    for f in file_list:
        with open(f, 'r') as f_:
            data = json.load(f_)

        for subject in subjects:
            if subject == 'Travis':
                continue # Discard unused subject
		
            if subject not in output:
                output[subject] = {}
                output_2d_det[subject] = {}
            for item in data:
                if 'mocap' not in item.keys():
                    continue

                if item['subject'] == subject:
                    canonical_name = item['movement']
                    canonical_name = canonical_name.replace('R_Single_Leg_Squat','Single_Leg_Squat_R') #fix order
                    canonical_name = canonical_name.replace('L_Single_Leg_Squat','Single_Leg_Squat_L') #fix order
                    canonical_name = canonical_name.replace('Squat_Jump', 'Jump_Squat') #To filter easier from Squat

                    #Filter by certain movements
                    keep_sample = False
                    for fm in filter_movements:
                        if fm in canonical_name:
                            keep_sample = True

                    if not keep_sample:
                        continue

                    #Downsample 3D keypoints and GRF to RGB camera rate
                    total_frames = min(item['total_frames'], len(item['frames'])) #T frames
                    mocap        = item['mocap']
                    grf          = item['grf']
                    frames       = item['frames']
                    
                    #Downsample GRF
                    time = grf['time']
                    indices = np.linspace(0, len(time)-1, total_frames).astype(np.int32)
                    time_sampled = np.array(time)[indices]
                    fx1 = np.array(grf['ground_force1_vx'])[indices]
                    fy1 = np.array(grf['ground_force1_vy'])[indices]
                    fz1 = np.array(grf['ground_force1_vz'])[indices]
                    fx2 = np.array(grf['ground_force2_vx'])[indices]
                    fy2 = np.array(grf['ground_force2_vy'])[indices]
                    fz2 = np.array(grf['ground_force2_vz'])[indices]
                    forces = np.stack((fx1,fy1,fz1,fx2,fy2,fz2), axis=1) #(T x 6)

                    #Normalize forces by subject mass
                    mass = subject_masses[subject]
                    forces /= mass

                    #Downsample 3D mocap markers
                    positions = []

                    for marker in mocap_markers:
                        positions.append(np.asarray(mocap[marker]))

                    #Gather 2D predictions from each view, not all trials have all views
                    keypoints_2d = []
                    keypoints_3d_tri = [] #Triangulated 3D keypoints from 2D views
                    view_lens = []
                    for cam in cameras:
                        kpts = [] 
                        min_len = 0
                        for idx, f in enumerate(frames):
                            if cam not in f: #Camera view is missing for this frame
                                continue
                            else: 
                                min_len += 1
                                if len(f[cam]) == 0: #Incase of missed detection, use previous pose
                                    kpts.append(frames[idx-1][cam]['keypoints'])
                                else:
                                    kpts.append(f[cam]['keypoints'])

                        if len(kpts) > 0:
                            kpts = np.array(kpts).reshape(-1, 17, 3)[...,:2] #Keep only x,y positions
                            keypoints_2d.append(kpts)
                            view_lens.append(min_len)

                    keypoints_3d_tri = np.stack([np.array(f['triangulated_pose']) for f in frames]) #Triangulated 3D keypoints from 2D views
                    positions = np.stack(positions, axis=1) #Expected shape: (T_p x 47 x 3) T_p should be exactly 1/6 sampling rate of force plates
                    indices   = np.linspace(0, len(time)/6-1, total_frames).astype(np.int32)
                    positions = positions[indices]

                    #Truncate to min_len across all viewpoints
                    min_len = min(view_lens)
                    keypoints_3d_tri = keypoints_3d_tri[:min_len]
                    positions        = positions[:min_len]
                    forces           = forces[:min_len]
                    for idx in range(len(keypoints_2d)):
                        keypoints_2d[idx] = keypoints_2d[idx][:min_len]

                    #positions /= 1000 # Supposed to originally be in meters, but scale seems to large. Reduce scale by 1000
                    #keypoints_3d_tri /= 1000 
                    output[subject][canonical_name] = {'positions':positions.astype('float32'), 'forces':forces, 'positions_triangulated':keypoints_3d_tri.astype('float32')}
                    output_2d_det[subject][canonical_name] = keypoints_2d
    
    print('Saving '+output_filename)
    np.savez_compressed(output_filename, data=output)
    
    print('Done.')
    
    #Create 2D detections pose file
    print('')
    print('Saving detection 2D poses...')

    print('Saving '+output_filename_2d_det)
    np.savez_compressed(output_filename_2d_det, positions_2d=output_2d_det, cam_names=cameras, metadata=coco_metadata)
    
    print('Done.')

    # Create 2D pose file
    print('')
    print('Computing ground-truth 2D poses...')
    dataset = ForcePoseDataset(output_filename + '.npz')
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
