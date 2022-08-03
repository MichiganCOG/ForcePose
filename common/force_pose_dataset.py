# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import numpy as np
import os
import json
import copy
from common.skeleton import Skeleton
from common.mocap_dataset import MocapDataset
from common.camera import normalize_screen_coordinates, image_coordinates
       
from scipy.spatial.transform import Rotation as R

mscoco_skeleton = Skeleton(parents=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,
                                    12, 13, 14, 15, 16],
       joints_left=[1, 3, 5, 7, 9, 11, 13, 15],
       joints_right=[2, 4, 6, 8, 10, 12, 14, 16])

camId2camName = {
        'aca4fbba-5dce-4812-a492-8ef72181ee02':'cam_17364068',
        '69a01acf-e2c3-4a25-ab22-c821a5110397':'cam_17400877',
        '175752c4-65ad-4bf5-857b-4d376c1a81e9':'cam_17400878',
        '40e02bae-36ac-4bb4-9987-2ba74d39a43c':'cam_17400879',
        '133ecbea-0edf-4007-9ed4-17f4577b00b9':'cam_17400880',
        '87838dab-243f-4bb3-8eb2-d3ded07b8447':'cam_17400881',
        '6d82e8be-a5af-4aca-889c-e09423de48e2':'cam_17400883',
        '204deb1c-a36e-4cca-b3ff-c9f3f02e2630':'cam_17400884',
        }

root_ext_file = './data/force_pose'
intr_file     = os.path.join(root_ext_file, 'FLIR_Intrinsics.inconfig')
ext_subj      = {'Subject1':'6_2_21.exconfig',
                 'Subject2':'6_2_21.exconfig',
                 'Subject3':'6_3_21.exconfig',
                 'Subject4':'6_3_21.exconfig',
                 'Subject6':'6_2_21.exconfig',
                 'Subject5':'6_3_21.exconfig',
                 'Subject7':'6_1_21.exconfig',
                 'Subject8':'6_1_21.exconfig'}

default_camera_params = [
        {'id': 'cam_17400878', 'res_w': 808, 'res_h': 608, 'intrinsic': np.array([0., 0., 0., 0., 0., 0., 0., 0., 0.], dtype='float32'), 'orientation': np.array([-0.03877663, -0.40723816,  0.9122225 , -0.02243944], dtype='float32'), 'translation': np.array([ -75.55734,  441.26627, 2764.9915], dtype='float32')}, #cam 1 
        {'id': 'cam_17400879', 'res_w': 808, 'res_h': 608, 'intrinsic': np.array([0., 0., 0., 0., 0., 0., 0., 0., 0.], dtype='float32'), 'orientation': np.array([ 0.72717106,  0.119923  , -0.6627182 , -0.13283576], dtype='float32'), 'translation': np.array([-129.12964, 1337.0112, 3203.659], dtype='float32')}, #cam 2
        {'id': 'cam_17400877', 'res_w': 808, 'res_h': 608, 'intrinsic': np.array([0., 0., 0., 0., 0., 0., 0., 0., 0.], dtype='float32'), 'orientation': np.array([ 0.9522375 , -0.03231593, -0.02971409, -0.3021862 ], dtype='float32'), 'translation': np.array([  87.427864,  677.8295, 3190.618], dtype='float32')}, #cam 3
        {'id': 'cam_17400881', 'res_w': 808, 'res_h': 608, 'intrinsic': np.array([0., 0., 0., 0., 0., 0., 0., 0., 0.], dtype='float32'), 'orientation': np.array([ 0.67822343, -0.19716202,  0.67604333, -0.2100132 ], dtype='float32'), 'translation': np.array([ -90.25427,  699.5648, 3324.8303], dtype='float32')}, #cam 4
        {'id': 'cam_17400880', 'res_w': 808, 'res_h': 608, 'intrinsic': np.array([0., 0., 0., 0., 0., 0., 0., 0., 0.], dtype='float32'), 'orientation': np.array([-0.47256616, -0.01741976,  0.88091624, -0.01908916], dtype='float32'), 'translation': np.array([  -7.431052, 1478.8634, 3149.1], dtype='float32')}, #cam 5
        {'id': 'cam_17400884', 'res_w': 808, 'res_h': 608, 'intrinsic': np.array([0., 0., 0., 0., 0., 0., 0., 0., 0.], dtype='float32'), 'orientation': np.array([ 0.9418704 ,  0.05173323, -0.33191526, -0.00600502], dtype='float32'), 'translation': np.array([-583.28436, 1241.7908, 3470.6575], dtype='float32')}, #cam 6
        {'id': 'cam_17364068', 'res_w': 808, 'res_h': 608, 'intrinsic': np.array([0., 0., 0., 0., 0., 0., 0., 0., 0.], dtype='float32'), 'orientation': np.array([ 0.94495755,  0.01065426,  0.32365692, -0.04677554], dtype='float32'), 'translation': np.array([ 318.34897, 1339.1471, 3470.3323], dtype='float32')}, #cam 7
        {'id': 'cam_17400883', 'res_w': 808, 'res_h': 608, 'intrinsic': np.array([0., 0., 0., 0., 0., 0., 0., 0., 0.], dtype='float32'), 'orientation': np.array([ 0.4621369 , -0.08820044,  0.8802112 , -0.0622769 ], dtype='float32'), 'translation': np.array([ -58.790325, 1041.3092, 3188.4736], dtype='float32')}  #cam 8
]

custom_camera_params = {
    'id': None,
    'res_w': 808, # Pulled from metadata
    'res_h': 608, # Pulled from metadata
    
    # Dummy camera parameters (taken from Human3.6M), only for visualization purposes
    'azimuth': 70, # Only used for visualization
    'orientation': [0.1407056450843811, -0.1500701755285263, -0.755240797996521, 0.6223280429840088],
    'translation': [1841.1070556640625, 4955.28466796875, 1563.4454345703125],
}

class ForcePoseDataset(MocapDataset):
    def __init__(self, detections_path, remove_static_joints=False):
        super().__init__(fps=50, skeleton=mscoco_skeleton)        
        
        # Load serialized dataset
        data = np.load(detections_path, allow_pickle=True)
        resolutions = data['data'].item()
        
        self._cameras = {}
        self._data = {}

        #Read intrinsics from all cameras
        with open(intr_file, 'r') as f:
            intr_data = json.load(f)

        for subject in resolutions.keys():
            self._data[subject] = {}
            
            #Read extrinsics from subject recording
            with open(os.path.join(root_ext_file, ext_subj[subject]), 'r') as f:
                ext_data = json.load(f)

            cams = []
            for i, (view, val) in enumerate(ext_data['views'].items()):
                cam = {}
                cam.update(default_camera_params[i])

                cam_name = camId2camName[view]
                cam['id'] = cam_name
                cam['res_w'] = 808
                cam['res_h'] = 608

                if subject == 'Subject6' and cam_name == 'cam_17400881': #Subject6 missing cam_17400881 
                    continue

                #Dummy intrinsic parameters. May not be used
                cam['intrinsic'] = np.zeros((9), dtype='float32')

                #Convert rotation matrix to quaternion, which it seems is what is used
                try:
                    orientation = R.from_matrix(val['matrices']['r']).as_quat()
                    cam['orientation'] = np.array(orientation, dtype='float32')
                    cam['translation'] = np.array(val['matrices']['t'], dtype='float32').squeeze()
                    #cam['translation'] = cam['translation']/1000 # mm to meters
                except TypeError:
                    print('{} - {} extrinsics not found. Using previous defaults'.format(subject, cam_name))

                cams.append(cam)

            self._cameras[subject] = cams
            
            for video_name, vid_dat in resolutions[subject].items():
                self._data[subject][video_name] = {'cameras': self._cameras[subject],\
                                                   'positions':vid_dat['positions'],\
                                                   'forces':vid_dat['forces'],\
                                                   'positions_triangulated':vid_dat['positions_triangulated']}
                
        if remove_static_joints: #Unused for now. Using MSCOCO (17) keypoints for triangulation
            # Bring the skeleton to 17 joints instead of the original 32
            self.remove_joints([4, 5, 9, 10, 11, 16, 20, 21, 22, 23, 24, 28, 29, 30, 31])
            
            # Rewire shoulders to the correct parents
            self._skeleton._parents[11] = 8
            self._skeleton._parents[14] = 8
            
    def supports_semi_supervised(self):
        return False
   
