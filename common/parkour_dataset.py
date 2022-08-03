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

parkour_skeleton = Skeleton(parents=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,
                                    12, 13, 14, 15],
       joints_left=[0, 5, 6, 7, 8, 13, 14, 15],
       joints_right=[1, 2, 3, 4, 9, 10, 11, 12])

default_camera_params = {
    'id': None,
    'res_w': 720, # Pulled from metadata
    'res_h': 576, # Pulled from metadata
    
    # Dummy camera parameters (taken from Human3.6M), only for visualization purposes
    'azimuth': 70, # Only used for visualization
    'orientation': np.array([0, 0, 0, 1], dtype='float32'),
    'translation': np.array([0, 0, 0], dtype='float32'),
}

#Subjects: 01, 02, 03, 04, 05, 06, 07
class ParkourDataset(MocapDataset):
    def __init__(self, detections_path, remove_static_joints=False):
        super().__init__(fps=50, skeleton=parkour_skeleton)        
        
        # Load serialized dataset
        data = np.load(detections_path, allow_pickle=True)
        resolutions = data['data'].item()
        
        self._cameras = {}
        self._data = {}

        for subject in resolutions.keys():
            self._data[subject] = {}
            
            cam = {}
            cam.update(default_camera_params)

            cam['id']    = 'cam_000001'
            cam['res_w'] = 720
            cam['res_h'] = 576

            #Dummy intrinsic parameters. May not be used
            cam['intrinsic'] = np.zeros((9), dtype='float32')

            self._cameras[subject] = [cam] #only one camera
            
            for video_name, vid_dat in resolutions[subject].items():
                self._data[subject][video_name] = {'cameras': self._cameras[subject],\
                                                   'positions':vid_dat['positions'],\
                                                   'forces':vid_dat['forces'],\
                                                   'seq_name':vid_dat['seq_name']}
                
        if remove_static_joints: #Unused for now. Using MSCOCO (17) keypoints for triangulation
            # Bring the skeleton to 17 joints instead of the original 32
            self.remove_joints([4, 5, 9, 10, 11, 16, 20, 21, 22, 23, 24, 28, 29, 30, 31])
            
            # Rewire shoulders to the correct parents
            self._skeleton._parents[11] = 8
            self._skeleton._parents[14] = 8
            
    def supports_semi_supervised(self):
        return False
   
