# -*- coding: utf-8 -*-
"""
Created on Thu Mar 31 17:16:07 2022

@author: asligar
"""

from bvh import Bvh
from scipy.spatial.transform import Rotation
import numpy as np
filename = './example_dae/untitled.bvh'
filename = './example_dae/people_walk_n_stand_04052022.bvh'
with open(filename) as f:
    mocap = Bvh(f.read())

print(next(mocap.root.filter('ROOT'))['OFFSET'])

joint_names= mocap.get_joints_names()
all_frames = []
all_frames2 = []
def recurse_joins():
    pass

for frame in range(mocap.nframes):
    all_joints_dict = {}
    all_joints_dict2 = {}
    for joint in joint_names:
        print(mocap.joint_parent_index(joint))
        joint_dict = {}
        joint_dict2 = {}
        channels = mocap.joint_channels(joint)
        for ch in channels:
            joint_dict[ch] = mocap.frame_joint_channel(frame, joint, ch)
            transform4x4 = np.eye(4)
            x_rot = 0;y_rot = 0;z_rot =0
            x_pos = 0;y_pos=0; z_pos = 0
            if ch == "Xrotation":
                x_rot = mocap.frame_joint_channel(frame, joint, ch)
            elif ch == "Yrotation":
                y_rot = mocap.frame_joint_channel(frame, joint, ch)
            elif ch == "Zrotation":
                z_rot = mocap.frame_joint_channel(frame, joint, ch)
            elif ch == "Xposition":
                x_pos = mocap.frame_joint_channel(frame, joint, ch)
            elif ch == "Yposition":
                y_pos = mocap.frame_joint_channel(frame, joint, ch)
            elif ch == "Zposition":
                z_pos = mocap.frame_joint_channel(frame, joint, ch)
        rot_temp = Rotation.from_euler('ZXY', [x_rot,y_rot,z_rot])
        transform4x4[0:3,3] = [x_pos,y_pos,z_pos]
        transform4x4[0:3,0:3] = rot_temp.as_matrix()
        all_joints_dict[joint] = joint_dict
        all_joints_dict2[joint] = transform4x4
    all_frames.append(all_joints_dict)
    all_frames2.append(all_joints_dict2)
