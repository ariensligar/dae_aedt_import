# -*- coding: utf-8 -*-
"""
Created on Mon Nov 22 11:46:09 2021

imports DAE files and creates mesh (STL files). Outputs mesh transforms versus
time.

@author: drey (original dae import and CS)
@author: asligar (modified dae import to support AEDT needs)
"""

import collada
import scipy
import scipy.interpolate
import numpy as np
import copy
import pyvista as pv
import os
from scipy.spatial.transform import Rotation
import glob
from difflib import SequenceMatcher


#utilitiy function for conversions between rot and euler
def rot_to_euler(rot, order='zyz',deg=True):
    rot = Rotation.from_matrix(rot)
    euler_angles = rot.as_euler(order,degrees=deg)
    #euler_angles = rot.as_rotvec(degrees=deg)
    return euler_angles
def euler_to_rot(euler, order='zyz',deg=True):
    
    if isinstance(euler,list):
        euler=np.array(euler)
    elif isinstance(euler, np.ndarray):
        if np.ndim(euler)==2:
            rot = np.zeros((euler.shape[0],3,3))
            for n in range(euler.shape[0]):
                rot_temp = Rotation.from_euler(order, euler[n],
                                                       degrees=deg)
                rot_temp = rot_temp.as_matrix()
                rot[n]=rot_temp
    else:
        rot = Rotation.from_euler(order,euler, degrees=deg)
        rot = rot.as_matrix()
    return rot

class CoordSys:
    def __init__(self):
    
        self.time = 0
        self.rot = np.eye(3)
        self.pos = np.zeros(3)
        self.transforms = None
        self.euler_transforms = None

      
        
    def __update(self,time):
        if self.transforms is None:
            self.rot = np.asarray(self.rot)
            self.pos = np.asarray(self.pos)
        else:
            # set from interpolated transform and estimate velocity
            self.set(self.transforms(time))
    
    def update(self,inGlobal = True,time=None):
        if time!=None:
          self.__update(time)
        if inGlobal:
            self.rot = np.ascontiguousarray(self.rot,dtype=np.float64)
            self.pos = np.ascontiguousarray(self.pos,dtype=np.float64)
    
    
    
    # return 4x4 matrix transform
    @property
    def transform4x4(self):
        ret = np.concatenate((np.asarray(self.rot),np.asarray(self.pos).reshape((-1,1))),axis=1)
        ret = np.concatenate((ret,np.array([[0,0,0,1]])),axis=0)
        return ret
    # set position/orientation from 4x4 transform
    def set(self,transform4x4):
        self.pos = transform4x4[0:3,3]
        self.rot = transform4x4[0:3,0:3]

    
    
class AnimatedDAE:
    def __init__(self,filename,save_path = './'):
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        self.save_path = save_path
        self.rot = np.eye(3)
        self.pos = np.zeros(3)
        self.filename = filename
        self.dae = collada.Collada(filename)


        if self.dae.assetInfo.upaxis == 'Y_UP':
            self.local2global = np.asarray(
              [[ 0, 0,+1, 0],\
              [+1, 0, 0, 0],\
              [ 0,+1, 0, 0],\
              [ 0, 0, 0,+1]])
        elif self.dae.assetInfo.upaxis == 'X_UP':
            self.local2global = np.asarray(
              [[ 0, +1,0, 0],\
              [0, 0, +1, 0],\
              [ +1,0, 0, 0],\
              [ 0, 0, 0,+1]])
        elif self.dae.assetInfo.upaxis == 'Z_UP':
            self.local2global = np.asarray(
              [[ +1, 0,+1, 0],\
              [0, +1, 0, 0],\
              [ 0,0, +1, 0],\
              [ 0, 0, 0,+1]])

        else: raise RuntimeError("unsupported up-axis in dae file")
        global2local = np.linalg.inv(self.local2global)
        # rescale to meters
        units2meters = self.dae.assetInfo.unitmeter
        self.rescaleTransform = np.ones((4,4))
        self.rescaleTransform[0:3,3] = units2meters
        color = [0.8,0.8,0.8]
        # build joint transform interpolations
        jointTransforms = {}


        self.clipLength = -1
        for anim in self.dae.animations:
            times = anim.sourceById[anim.name + "-Matrix-animation-input"]
            times = np.reshape(times,-1)
            self.times = times
            self.clipLength = max(self.clipLength,times[-1])
            transforms = anim.sourceById[anim.name + "-Matrix-animation-output-transform"]
            transforms = np.reshape(transforms,(-1,4,4)) 
            transforms = transforms*self.rescaleTransform
            
            # HFSS requires euler angles for object orientations, the 3x3 rotational
            # matrix that is defined here, needs to eventually be converted to 
            # euler angles.
            # I found that if I use the rotation matrix, and create an iterpolation function
            # from it. When I convert to euler
            # angles, then unwrap the phase. I get interpolation errors.
            # What seems to work better, is convert rotational matrix to euler
            # then unwrap, the convert back to rotation matrix. then when I 
            # convert it back to euler and unwrap I don't get interpolation errors
            
            transforms2 = copy.deepcopy(transforms)
            rot_3x3 = transforms[:,0:3,0:3]
            euler_angs = rot_to_euler(rot_3x3,order='zyz',deg=False)

            phi = np.rad2deg(np.unwrap(euler_angs[:,0],period=np.pi*2))
            theta = np.rad2deg(np.unwrap(euler_angs[:,1],period=np.pi*2))
            psi = np.rad2deg(np.unwrap(euler_angs[:,2],period=np.pi*2))
            euler = np.vstack((phi,theta,psi)).T
            
            rot = euler_to_rot(euler)
            transforms2[:,0:3,0:3] = rot
            interpFunc = scipy.interpolate.interp1d(times,transforms2,axis=0,assume_sorted=True)
            jointTransforms[anim.name] = interpFunc
            

        def recurseJoints(
            sceneTree = None,
            nodeID = None):
              
            if sceneTree is None:
                nodeID = "root"
                
                nodeCS = CoordSys()
                nodeCS.set(self.local2global)
                nodeCS.update(time=0.,inGlobal=False)
                sceneTree = {}
                children = [self.dae.scene.nodes[0]]
                sceneTree[nodeID] = (nodeCS,children)
            (nodeCS,children) = sceneTree[nodeID]
            for child in children:
                childID = child.id
            
                childCS = CoordSys()
                if childID in jointTransforms:
                    childCS.transforms = jointTransforms[childID]

                else:
                    childTransform = child.matrix*self.rescaleTransform
                    childCS.set(childTransform)
                childCS.update(time=0.,inGlobal=False)
                sceneTree[childID] = (childCS,child.children)
                recurseJoints(sceneTree,childID)
  
            return sceneTree
        self.sceneTree = recurseJoints()
        # split mesh into connected components
        self.subMeshes = {}

        meshes = []
        for skin in self.dae.controllers:
            geometry = skin.geometry
            primitive = geometry.primitives[0]
            verts = np.asarray(primitive.vertex)
            tris  = np.asarray(primitive.vertex_index).reshape((-1,3))
            #for pyvista, needs number of points as first dimension
            num_points_connected = np.ones((len(tris),1))*3
            face_indices=np.hstack((num_points_connected,tris))
            face_indices = face_indices.astype(int)
            N = primitive.vertex.shape[0]
            rowIdxs = np.concatenate((tris[:,0],tris[:,1],tris[:,2]),axis=0)
            colIdxs = np.concatenate((tris[:,1],tris[:,2],tris[:,0]),axis=0)
            data = np.ones(np.shape(rowIdxs))
            graph = scipy.sparse.coo_matrix((data, (rowIdxs, colIdxs)), shape=(N, N))
            graph = scipy.sparse.csr_matrix(graph)
            (numComponents, labels) = scipy.sparse.csgraph.connected_components(
              csgraph=graph, directed=False, return_labels=True)
            
            for iSubMesh in range(numComponents):
                # partition into submesh by connected components
                vIdxs = np.reshape(np.where(labels == iSubMesh),-1)
                subVerts = verts[vIdxs]
                triMask = np.zeros(np.shape(tris)[0])
                

                
                for vIdx in vIdxs:
                    triMask = np.logical_or(triMask,np.any(tris == vIdx,axis=1))
                reindexMap = dict(zip(vIdxs,np.asarray(range(len(vIdxs)))))
                subTris = np.vectorize(reindexMap.get)(tris[triMask])
                
                num_points_connected = np.ones((len(subTris),1))*3
                face_indices=np.hstack((num_points_connected,subTris))
                face_indices = face_indices.astype(int)
                
                # tabulate all joints that contribute to this submesh
                jIdxs = {}
                for vIdx in vIdxs:
                    for iJoint,jIdx in enumerate(skin.joint_index[vIdx]):
                        wIdxs = skin.weight_index[vIdx]
                        wIdx = wIdxs[iJoint]
                        weight = skin.weights[wIdx][0]/len(vIdxs)
                        if jIdx in jIdxs:
                            jIdxs[jIdx] = jIdxs[jIdx]+weight
                        else:
                            jIdxs[jIdx] = weight
                # associate submesh with node for joint that has maximum weight
                jIdx = max(jIdxs, key=jIdxs.get)
                (nodeCS,_) = self.sceneTree[skin.weight_joints[jIdx]]
                
                # undo bind pose for this submesh
                jInv = np.asarray(skin.joint_matrices[skin.weight_joints[jIdx]])
                jInv = copy.deepcopy(jInv)
                jInv[0:3,3] *= units2meters
                for vIdx in range(np.shape(subVerts)[0]):
                    v = np.ones(4)
                    v[0:3] = units2meters*subVerts[vIdx,:]
                    v = np.dot(jInv,v)
                    subVerts[vIdx] = v[0:3]
                    
                vData = np.ascontiguousarray(subVerts,dtype=np.float32).flatten()
                iData = np.ascontiguousarray(subTris,dtype=np.int32).flatten()
                mesh = pv.PolyData(subVerts,face_indices)
                
                file_name = self.save_path + skin.weight_joints[jIdx] + ".stl"
                mesh.save(file_name, binary=True)
            
                self.subMeshes[skin.weight_joints[jIdx]]=mesh
                temp_dict = {'file_name':file_name,'mesh':mesh}

    
    def __updateGlobalJointTransforms(
        self,
        time,
        jointTransforms={},
        nodeID="root",
        parentID=None):
        
        global maxErrors
        (nodeCS,children) = self.sceneTree[nodeID]
        if parentID is None:
            parentTransform=np.eye(4)
        else:
            parentTransform = jointTransforms[parentID]
        
        nodeCS.pos = self.pos
        nodeCS.rot = self.rot
        #deal with loop
        time = time%self.clipLength
        
        nodeCS.update(time=time,inGlobal=False)
        nodeTransform = np.dot(parentTransform,nodeCS.transform4x4)
        
        jointTransforms[nodeID] = nodeTransform


        for child in children:
            self.__updateGlobalJointTransforms(time,jointTransforms,child.id,nodeID)
        return jointTransforms
    
    
    
      # update the rigid body approximation of joint-weighted meshes used by the RSS
    def updateRigidMeshes(self,time):
        
        transforms = self.__updateGlobalJointTransforms(time)
        mesh_dict = {}

        for nodeID in self.sceneTree:
            if nodeID in self.subMeshes:
                
                file_name = self.save_path + nodeID + ".stl"
                mesh = self.subMeshes[nodeID]
                #doing all mesh transformations in main script so commented this out
                #mesh.transform(transforms[nodeID])
                temp_dict = {'file_name':file_name,'mesh':mesh,'transform':transforms[nodeID]}
                mesh_dict[nodeID] = temp_dict

        return mesh_dict
    
class AnimatedDAE2:
    '''
    testing dae importt from https://mocap.cs.sfu.ca/
    
    There are some difficult names that are not consistent between sources
    of dae files. So for this I try my best to correct, but you might have to 
    change some hard coded values
    
    '''
    def __init__(self,filename,save_path = './'):
        
        self.all_possible_files = glob.glob('meshes/mixamorig_*.stl')
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        self.save_path = save_path
        self.rot = np.eye(3)
        self.pos = np.zeros(3)
        self.filename = filename
        self.dae = collada.Collada(filename)


        if self.dae.assetInfo.upaxis == 'Y_UP':
            self.local2global = np.asarray(
              [[ 0, 0,+1, 0],\
              [+1, 0, 0, 0],\
              [ 0,+1, 0, 0],\
              [ 0, 0, 0,+1]])
        elif self.dae.assetInfo.upaxis == 'X_UP':
            self.local2global = np.asarray(
              [[ 0, +1,0, 0],\
              [0, 0, +1, 0],\
              [ +1,0, 0, 0],\
              [ 0, 0, 0,+1]])
        elif self.dae.assetInfo.upaxis == 'Z_UP':
            self.local2global = np.asarray(
              [[ +1, 0,+1, 0],\
              [0, +1, 0, 0],\
              [ 0,0, +1, 0],\
              [ 0, 0, 0,+1]])

        else: raise RuntimeError("unsupported up-axis in dae file")
        global2local = np.linalg.inv(self.local2global)
        # rescale to meters
        units2meters = self.dae.assetInfo.unitmeter
        self.rescaleTransform = np.ones((4,4))
        self.rescaleTransform[0:3,3] = units2meters
        color = [0.8,0.8,0.8]
        # build joint transform interpolations
        jointTransforms = {}


        # for anim in self.dae.animations:
        #     times = anim.sourceById[anim.name + "-Matrix-animation-input"]
        #     times = np.reshape(times,-1)
        #     self.times = times
        #     self.clipLength = max(self.clipLength,times[-1])
        #     transforms = anim.sourceById[anim.name + "-Matrix-animation-output-transform"]


        self.clipLength = -1
        child_names_by_id = {}
        scene = self.dae.scenes[0].nodes[0]
        


            
        for anim in self.dae.animations:
            name = list(anim.sourceById.keys())
            times = anim.sourceById[name[0]]
            times = np.reshape(times,-1)
            times = times-np.min(times) #zero shift
            self.times = times
            self.clipLength = max(self.clipLength,times[-1])

        #get common string in animation names for splitting string later
        all_part_names = []
        for child in anim.children:
            all_part_names.append(child.id)

        string2 = all_part_names[0]
        all_matches = []
        for i in range(1, len(all_part_names)):
            string1 = string2
            string2 = all_part_names[i]
            match = SequenceMatcher(None, string1, string2).find_longest_match(0, len(string1), 0, len(string2))
            all_matches.append(string1[match.a: match.a + match.size])
        shortest_string = ''.zfill(255)
        for each in all_matches:
            if len(each)<len(shortest_string):
                shortest_string = each
        common_str = shortest_string

        #assuming only one animation, anim
        for child in anim.children:

            transforms = anim.sourceById[child.id + "-output"]
            transforms = np.reshape(transforms,(-1,4,4)) 
            transforms = transforms*self.rescaleTransform
            
            # HFSS requires euler angles for object orientations, the 3x3 rotational
            # matrix that is defined here, needs to eventually be converted to 
            # euler angles.
            # I found that if I use the rotation matrix, and create an iterpolation function
            # from it. When I convert to euler
            # angles, then unwrap the phase. I get interpolation errors.
            # What seems to work better, is convert rotational matrix to euler
            # then unwrap, the convert back to rotation matrix. then when I 
            # convert it back to euler and unwrap I don't get interpolation errors
            
            transforms2 = copy.deepcopy(transforms)
            rot_3x3 = transforms[:,0:3,0:3]
            euler_angs = rot_to_euler(rot_3x3,order='zyz',deg=False)

            phi = np.rad2deg(np.unwrap(euler_angs[:,0],period=np.pi*2))
            theta = np.rad2deg(np.unwrap(euler_angs[:,1],period=np.pi*2))
            psi = np.rad2deg(np.unwrap(euler_angs[:,2],period=np.pi*2))
            euler = np.vstack((phi,theta,psi)).T
            
            rot = euler_to_rot(euler)
            transforms2[:,0:3,0:3] = rot
            interpFunc = scipy.interpolate.interp1d(times,transforms2,axis=0,assume_sorted=True)
            #temporary renmaing until I figure out easier and more robust way to get names
            #sometimes the _pos_matrix exists, sometimes it doesn't
            childname =child.id.split(common_str)[-1].split('_pose_matrix')[0]#.replace('_',':')

            if childname=='transform':
                childname='sbui_Hips'


            jointTransforms[childname] = interpFunc
            child_names_by_id[child.id] = childname
        def recurseJoints(
            sceneTree = None,
            nodeID = None):
              
            
            if sceneTree is None:
                nodeID = "root"
                #root_name = 
                nodeCS = CoordSys()
                nodeCS.set(self.local2global)
                nodeCS.update(time=0.,inGlobal=False)
                sceneTree = {}

                sceneTree[nodeID] = (nodeCS,scene.children)
            (nodeCS,children) = sceneTree[nodeID]
            for child in children:
                if hasattr(child, 'id'):
                    childID = child.id
                    if childID !='sbui_Hips':
                        childID = childID.replace('sbui_Hips_','')
                    #print(childID)
                    childCS = CoordSys()
                    if childID in jointTransforms:
                        childCS.transforms = jointTransforms[childID]
    
                    else:
                        childTransform = child.matrix*self.rescaleTransform
                        childCS.set(childTransform)
                    childCS.update(time=0.,inGlobal=True)
                    sceneTree[childID] = (childCS,child.children)
                    recurseJoints(sceneTree,childID)
            return sceneTree
        self.sceneTree = recurseJoints()
        # split mesh into connected components
        self.subMeshes = {}

        self.all_possible_files
        meshes = []
        self.mapped_stl_filenames = {}
        scale_factor = 1
        legs = ['sbui_LeftLeg','sbui_RightLeg','sbui_LeftUpLeg','sbui_RightUpLeg']
        foot = ['sbui_LeftFoot','sbui_RightFoot']
        larms = ['sbui_LeftArm','sbui_LeftForeArm']
        rarms = ['sbui_RightArm','sbui_RightForeArm']
        for n, part in enumerate(self.sceneTree):
            if (part !='root' and  part !='sbui_Spine'):
                # part_str = part.replace("_005_Walking001_0005_Walking001_","")
                # part_str = part_str.replace("_pose_matrix","")
                part_str = part.split("_")[-1]
                #print(part)
                for file in self.all_possible_files:
                    if part_str in file:
                        mesh = pv.read(file)
                        mesh.scale([scale_factor, scale_factor, scale_factor])
                        if part == 'sbui_Spine1':
                            mesh.translate([0,-.05,0])
                        if part in legs:
                            mesh.rotate_x(180)
                        if part in foot:
                            mesh.rotate_x(-90)
                            mesh.rotate_y(180)

                        if part in rarms:
                            mesh.rotate_y(-90)
                            mesh.rotate_x(90)
                        if part in larms:
                            mesh.rotate_x(180)
                            mesh.rotate_y(-90)
                            mesh.rotate_x(90)
                            
                            mesh.rotate_z(180)
                        self.mapped_stl_filenames[part] = file
                        self.subMeshes[part]=mesh
 

    
    def __updateGlobalJointTransforms(
        self,
        time,
        jointTransforms={},
        nodeID="root",
        parentID=None):
        
        global maxErrors
        (nodeCS,children) = self.sceneTree[nodeID]
        if parentID is None:
            parentTransform=np.eye(4)
        else:
            parentTransform = jointTransforms[parentID]

        nodeCS.pos = self.pos
        nodeCS.rot = self.rot
        #deal with loop
        time = time%self.clipLength
        
        nodeCS.update(time=time,inGlobal=True)
        nodeTransform = np.dot(parentTransform,nodeCS.transform4x4)
        #print(nodeCS.transform4x4)
        jointTransforms[nodeID] = nodeTransform


        for child in children:
            if hasattr(child,'id'):
                childID = child.id
                if childID !='sbui_Hips':
                    childID = childID.replace('sbui_Hips_','')
                self.__updateGlobalJointTransforms(time,jointTransforms,childID,nodeID)
        return jointTransforms
    
    
    
      # update the rigid body approximation of joint-weighted meshes used by the RSS
    def updateRigidMeshes(self,time):
        
        transforms = self.__updateGlobalJointTransforms(time)
        mesh_dict = {}

        for nodeID in self.sceneTree:
            if nodeID in self.subMeshes:
                #print(nodeID)
                #file_name = self.save_path + nodeID + ".stl"
                file_name = self.mapped_stl_filenames[nodeID]
                mesh = self.subMeshes[nodeID]
                #doing all mesh transformations in main script so commented this out
                #mesh.transform(transforms[nodeID])
                temp_dict = {'file_name':file_name,'mesh':mesh,'transform':transforms[nodeID]}
                mesh_dict[nodeID] = temp_dict

        return mesh_dict


    
class AnimatedDAE3:
    '''
    testing dae importt from https://mocap.cs.sfu.ca/
    
    https://www.autodesk.com/developer-network/platform-technologies/fbx-converter-archives
    
    There are some difficult names that are not consistent between sources
    of dae files. So for this I try my best to correct, but you might have to 
    change some hard coded values
    
    '''
    def __init__(self,filename,save_path = './',fixed_str = '__0005_Walking_x2d_Layer0_transform-'):
        
        self.all_possible_files = glob.glob('meshes/mixamorig_*.stl')
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        self.save_path = save_path
        self.rot = np.eye(3)
        self.pos = np.zeros(3)
        self.filename = filename
        self.dae = collada.Collada(filename)


        if self.dae.assetInfo.upaxis == 'Y_UP':
            self.local2global = np.asarray(
              [[ 0, 0,+1, 0],\
              [+1, 0, 0, 0],\
              [ 0,+1, 0, 0],\
              [ 0, 0, 0,+1]])
        elif self.dae.assetInfo.upaxis == 'X_UP':
            self.local2global = np.asarray(
              [[ 0, +1,0, 0],\
              [0, 0, +1, 0],\
              [ +1,0, 0, 0],\
              [ 0, 0, 0,+1]])
        elif self.dae.assetInfo.upaxis == 'Z_UP':
            self.local2global = np.asarray(
              [[ +1, 0,+1, 0],\
              [0, +1, 0, 0],\
              [ 0,0, +1, 0],\
              [ 0, 0, 0,+1]])

        else: raise RuntimeError("unsupported up-axis in dae file")
        global2local = np.linalg.inv(self.local2global)
        # rescale to meters
        units2meters = self.dae.assetInfo.unitmeter
        self.rescaleTransform = np.ones((4,4))
        self.rescaleTransform[0:3,3] = units2meters
        color = [0.8,0.8,0.8]
        # build joint transform interpolations
        jointTransforms = {}


        # for anim in self.dae.animations:
        #     times = anim.sourceById[anim.name + "-Matrix-animation-input"]
        #     times = np.reshape(times,-1)
        #     self.times = times
        #     self.clipLength = max(self.clipLength,times[-1])
        #     transforms = anim.sourceById[anim.name + "-Matrix-animation-output-transform"]


        self.clipLength = -1
        child_names_by_id = {}
        scene = self.dae.scenes[0].nodes[0]
        


            
        for anim in self.dae.animations:
            name = list(anim.sourceById.keys())
            times = anim.sourceById[name[0]]
            times = np.reshape(times,-1)
            times = times-np.min(times) #zero shift
            self.times = times
            self.clipLength = max(self.clipLength,times[-1])

        #get common string in animation names for splitting string later
        all_part_names = []
        for child in self.dae.animations:
            all_part_names.append(child.id)

        string2 = all_part_names[0]
        all_matches = []
        for i in range(1, len(all_part_names)):
            string1 = string2
            string2 = all_part_names[i]
            match = SequenceMatcher(None, string1, string2).find_longest_match(0, len(string1), 0, len(string2))
            all_matches.append(string1[match.a: match.a + match.size])
        shortest_string = ''.zfill(255)
        for each in all_matches:
            if len(each)<len(shortest_string):
                shortest_string = each
        common_str = shortest_string

        #assuming only one animation, anim
        for child in self.dae.animations:
            transform_keys = list(anim.sourceById.keys())
            transf_key = ''
            for each in transform_keys:
                if 'output' in each:
                    transf_key = each
                
            transforms = anim.sourceById[transf_key]
            transforms = np.reshape(transforms,(-1,4,4)) 
            transforms = transforms*self.rescaleTransform
            
            # HFSS requires euler angles for object orientations, the 3x3 rotational
            # matrix that is defined here, needs to eventually be converted to 
            # euler angles.
            # I found that if I use the rotation matrix, and create an iterpolation function
            # from it. When I convert to euler
            # angles, then unwrap the phase. I get interpolation errors.
            # What seems to work better, is convert rotational matrix to euler
            # then unwrap, the convert back to rotation matrix. then when I 
            # convert it back to euler and unwrap I don't get interpolation errors
            
            transforms2 = copy.deepcopy(transforms)
            rot_3x3 = transforms[:,0:3,0:3]
            euler_angs = rot_to_euler(rot_3x3,order='zyz',deg=False)

            phi = np.rad2deg(np.unwrap(euler_angs[:,0],period=np.pi*2))
            theta = np.rad2deg(np.unwrap(euler_angs[:,1],period=np.pi*2))
            psi = np.rad2deg(np.unwrap(euler_angs[:,2],period=np.pi*2))
            euler = np.vstack((phi,theta,psi)).T
            
            rot = euler_to_rot(euler)
            transforms2[:,0:3,0:3] = rot
            interpFunc = scipy.interpolate.interp1d(times,transforms2,axis=0,assume_sorted=True)
            #temporary renmaing until I figure out easier and more robust way to get names
            #sometimes the _pos_matrix exists, sometimes it doesn't
            childname =child.id.split(common_str)[-1].split('__0005_Walking_x2d_Layer0_transform-output')[0]#.replace('_',':')

            if childname=='transform':
                childname='sbui_Hips'


            jointTransforms[childname] = interpFunc
            child_names_by_id[child.id] = childname
        def recurseJoints(
            sceneTree = None,
            nodeID = None):
              
            
            if sceneTree is None:
                nodeID = "root"
                #root_name = 
                nodeCS = CoordSys()
                nodeCS.set(self.local2global)
                nodeCS.update(time=0.,inGlobal=False)
                sceneTree = {}

                sceneTree[nodeID] = (nodeCS,scene.children)
            (nodeCS,children) = sceneTree[nodeID]
            for child in children:
                if hasattr(child, 'id'):
                    childID = child.id
                    if childID !='sbui_Hips':
                        childID = childID.replace('Armature_','')
                    #print(childID)
                    childCS = CoordSys()
                    if childID in jointTransforms:
                        childCS.transforms = jointTransforms[childID]
    
                    else:
                        childTransform = child.matrix*self.rescaleTransform
                        childCS.set(childTransform)
                    childCS.update(time=0.,inGlobal=True)
                    sceneTree[childID] = (childCS,child.children)
                    recurseJoints(sceneTree,childID)
            return sceneTree
        self.sceneTree = recurseJoints()
        # split mesh into connected components
        self.subMeshes = {}

        self.all_possible_files
        meshes = []
        self.mapped_stl_filenames = {}
        scale_factor = 1
        legs = ['sbui_LeftLeg','sbui_RightLeg','sbui_LeftUpLeg','sbui_RightUpLeg']
        foot = ['sbui_LeftFoot','sbui_RightFoot']
        larms = ['sbui_LeftArm','sbui_LeftForeArm']
        rarms = ['sbui_RightArm','sbui_RightForeArm']
        for n, part in enumerate(self.sceneTree):
            if (part !='root' and  part !='sbui_Spine'):
                # part_str = part.replace("_005_Walking001_0005_Walking001_","")
                # part_str = part_str.replace("_pose_matrix","")
                part_str = part.split("_")[-1]
                #print(part)
                for file in self.all_possible_files:
                    if part_str in file:
                        mesh = pv.read(file)
                        mesh.scale([scale_factor, scale_factor, scale_factor])
                        if part == 'sbui_Spine1':
                            mesh.translate([0,-.05,0])
                        if part in legs:
                            mesh.rotate_x(180)
                        if part in foot:
                            mesh.rotate_x(-90)
                            mesh.rotate_y(180)

                        if part in rarms:
                            mesh.rotate_y(-90)
                            mesh.rotate_x(90)
                        if part in larms:
                            mesh.rotate_x(180)
                            mesh.rotate_y(-90)
                            mesh.rotate_x(90)
                            
                            mesh.rotate_z(180)
                        self.mapped_stl_filenames[part] = file
                        self.subMeshes[part]=mesh
 

    
    def __updateGlobalJointTransforms(
        self,
        time,
        jointTransforms={},
        nodeID="root",
        parentID=None):
        
        global maxErrors
        (nodeCS,children) = self.sceneTree[nodeID]
        if parentID is None:
            parentTransform=np.eye(4)
        else:
            parentTransform = jointTransforms[parentID]

        nodeCS.pos = self.pos
        nodeCS.rot = self.rot
        #deal with loop
        time = time%self.clipLength
        
        nodeCS.update(time=time,inGlobal=True)
        nodeTransform = np.dot(parentTransform,nodeCS.transform4x4)
        #print(nodeCS.transform4x4)
        jointTransforms[nodeID] = nodeTransform


        for child in children:
            if hasattr(child,'id'):
                childID = child.id
                if childID !='sbui_Hips':
                    childID = childID.replace('sbui_Hips_','')
                self.__updateGlobalJointTransforms(time,jointTransforms,childID,nodeID)
        return jointTransforms
    
    
    
      # update the rigid body approximation of joint-weighted meshes used by the RSS
    def updateRigidMeshes(self,time):
        
        transforms = self.__updateGlobalJointTransforms(time)
        mesh_dict = {}

        for nodeID in self.sceneTree:
            if nodeID in self.subMeshes:
                #print(nodeID)
                #file_name = self.save_path + nodeID + ".stl"
                file_name = self.mapped_stl_filenames[nodeID]
                mesh = self.subMeshes[nodeID]
                #doing all mesh transformations in main script so commented this out
                #mesh.transform(transforms[nodeID])
                temp_dict = {'file_name':file_name,'mesh':mesh,'transform':transforms[nodeID]}
                mesh_dict[nodeID] = temp_dict

        return mesh_dict
    
class AnimatedDAE4:
    '''
    testing dae importt from https://mocap.cs.sfu.ca/
    
    There are some difficult names that are not consistent between sources
    of dae files. So for this I try my best to correct, but you might have to 
    change some hard coded values
    
    '''
    def __init__(self,filename,save_path = './'):
        
        self.all_possible_files2 = glob.glob('meshes/*.stl')
        self.all_possible_files  = []
        for each in self.all_possible_files2 :
            if 'mixamorig' not in each:
                self.all_possible_files.append(each) 
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        self.save_path = save_path
        self.rot = np.eye(3)
        self.pos = np.zeros(3)
        self.filename = filename
        self.dae = collada.Collada(filename)


        if self.dae.assetInfo.upaxis == 'Y_UP':
            self.local2global = np.asarray(
              [[ 0, 0,+1, 0],\
              [+1, 0, 0, 0],\
              [ 0,+1, 0, 0],\
              [ 0, 0, 0,+1]])
        elif self.dae.assetInfo.upaxis == 'X_UP':
            self.local2global = np.asarray(
              [[ 0, +1,0, 0],\
              [0, 0, +1, 0],\
              [ +1,0, 0, 0],\
              [ 0, 0, 0,+1]])
        elif self.dae.assetInfo.upaxis == 'Z_UP':
            self.local2global = np.asarray(
              [[ +1, 0,+1, 0],\
              [0, +1, 0, 0],\
              [ 0,0, +1, 0],\
              [ 0, 0, 0,+1]])

        else: raise RuntimeError("unsupported up-axis in dae file")
        global2local = np.linalg.inv(self.local2global)
        # rescale to meters
        units2meters = self.dae.assetInfo.unitmeter
        self.rescaleTransform = np.ones((4,4))
        self.rescaleTransform[0:3,3] = units2meters*1e-2
        color = [0.8,0.8,0.8]
        # build joint transform interpolations
        jointTransforms = {}


        # for anim in self.dae.animations:
        #     times = anim.sourceById[anim.name + "-Matrix-animation-input"]
        #     times = np.reshape(times,-1)
        #     self.times = times
        #     self.clipLength = max(self.clipLength,times[-1])
        #     transforms = anim.sourceById[anim.name + "-Matrix-animation-output-transform"]


        self.clipLength = -1
        child_names_by_id = {}
        scene = self.dae.scenes[0].nodes[0]
        


            
        for anim in self.dae.animations:
            name = list(anim.sourceById.keys())
            times = anim.sourceById[name[0]]
            times = np.reshape(times,-1)
            times = times-np.min(times) #zero shift
            self.times = times
            self.clipLength = max(self.clipLength,times[-1])
        self.anim_name = anim.name
        #get common string in animation names for splitting string later
        all_part_names = []
        for child in anim.children:
            all_part_names.append(child.id)

        string2 = all_part_names[0]
        all_matches = []
        for i in range(1, len(all_part_names)):
            string1 = string2
            string2 = all_part_names[i]
            match = SequenceMatcher(None, string1, string2).find_longest_match(0, len(string1), 0, len(string2))
            all_matches.append(string1[match.a: match.a + match.size])
        shortest_string = ''.zfill(255)
        for each in all_matches:
            if len(each)<len(shortest_string):
                shortest_string = each
        common_str = shortest_string

        #assuming only one animation, anim
        for child in anim.children:

            transforms = anim.sourceById[child.id + "-output"]
            transforms = np.reshape(transforms,(-1,4,4)) 
            transforms = transforms*self.rescaleTransform
            
            # HFSS requires euler angles for object orientations, the 3x3 rotational
            # matrix that is defined here, needs to eventually be converted to 
            # euler angles.
            # I found that if I use the rotation matrix, and create an iterpolation function
            # from it. When I convert to euler
            # angles, then unwrap the phase. I get interpolation errors.
            # What seems to work better, is convert rotational matrix to euler
            # then unwrap, the convert back to rotation matrix. then when I 
            # convert it back to euler and unwrap I don't get interpolation errors
            
            transforms2 = copy.deepcopy(transforms)
            rot_3x3 = transforms[:,0:3,0:3]
            euler_angs = rot_to_euler(rot_3x3,order='zyz',deg=False)

            phi = np.rad2deg(np.unwrap(euler_angs[:,0],period=np.pi*2))
            theta = np.rad2deg(np.unwrap(euler_angs[:,1],period=np.pi*2))
            psi = np.rad2deg(np.unwrap(euler_angs[:,2],period=np.pi*2))
            euler = np.vstack((phi,theta,psi)).T
            
            rot = euler_to_rot(euler)
            transforms2[:,0:3,0:3] = rot
            interpFunc = scipy.interpolate.interp1d(times,transforms2,axis=0,assume_sorted=True)
            #temporary renmaing until I figure out easier and more robust way to get names
            #sometimes the _pos_matrix exists, sometimes it doesn't
            childname =child.id.split(common_str)[-1].split('_pose_matrix')[0]#.replace('_',':')

            # if childname=='transform':
            #     childname='sbui_Hips'


            jointTransforms[childname] = interpFunc
            child_names_by_id[child.id] = childname
        def recurseJoints(
            sceneTree = None,
            nodeID = None):
              
            
            if sceneTree is None:
                nodeID = "root"
                #root_name = 
                nodeCS = CoordSys()
                nodeCS.set(self.local2global)
                nodeCS.update(time=0.,inGlobal=False)
                sceneTree = {}

                sceneTree[nodeID] = (nodeCS,scene.children)
            (nodeCS,children) = sceneTree[nodeID]
            for child in children:
                if hasattr(child, 'id'):
                    childID = child.id
                    #if childID !='sbui_Hips':
                    childID = childID.replace(self.anim_name + '_','')
                    #print(childID)
                    childCS = CoordSys()
                    if childID in jointTransforms:
                        childCS.transforms = jointTransforms[childID]
    
                    else:
                        childTransform = child.matrix*self.rescaleTransform
                        childCS.set(childTransform)
                    childCS.update(time=0.,inGlobal=True)
                    sceneTree[childID] = (childCS,child.children)
                    recurseJoints(sceneTree,childID)
            return sceneTree
        self.sceneTree = recurseJoints()
        # split mesh into connected components
        self.subMeshes = {}

        self.all_possible_files
        meshes = []
        self.mapped_stl_filenames = {}
        scale_factor = 1
        llegs = ['UpperLeg_L','LowerLeg_L']
        rlegs = ['LowerLeg_R','UpperLeg_R']
        foot = ['Foot_L','Foot_R']
        larms = ['LowerArm_L','UpperArm_L']
        rarms = ['UpperArm_R','LowerArm_R']
        for n, part in enumerate(self.sceneTree):
            if (part !='joint_Root' and  part !='root'):
                # part_str = part.replace("_005_Walking001_0005_Walking001_","")
                # part_str = part_str.replace("_pose_matrix","")
                part_str = part.split("_")[-1]
                #print(part)
                
                for file in self.all_possible_files:
                    if part in file:
                        mesh = pv.read(file)
                        mesh.scale([scale_factor, scale_factor, scale_factor])
                        if part == 'Spine':
                            mesh.translate([0,-.0,0])
                            #mesh.rotate_z(180)
                            mesh.rotate_y(-90)
                        if part == 'Head':
                            #mesh.translate([0,-.05,0])
                            mesh.rotate_z(-90)
                            mesh.rotate_x(-90)
                        if part in llegs:
                            mesh.rotate_x(180)
                            mesh.rotate_y(180)
                            #mesh.translate([-.1,0,0])
                        if part in rlegs:
                            mesh.rotate_x(180)
                        if part in foot:
                            mesh.rotate_x(-90)
                            mesh.rotate_y(180)

                        if part in rarms:
                            mesh.rotate_y(-90)
                            mesh.rotate_x(90)
                        if part in larms:
                            mesh.rotate_x(180)
                            mesh.rotate_y(-90)
                            mesh.rotate_x(90)
                            
                            mesh.rotate_z(180)
                        self.mapped_stl_filenames[part] = file
                        self.subMeshes[part]=mesh
 

    
    def __updateGlobalJointTransforms(
        self,
        time,
        jointTransforms={},
        nodeID="root",
        parentID=None):
        
        global maxErrors
        (nodeCS,children) = self.sceneTree[nodeID]
        if parentID is None:
            parentTransform=np.eye(4)
        else:
            parentTransform = jointTransforms[parentID]

        nodeCS.pos = self.pos
        nodeCS.rot = self.rot
        #deal with loop
        time = time%self.clipLength
        
        nodeCS.update(time=time,inGlobal=True)
        nodeTransform = np.dot(parentTransform,nodeCS.transform4x4)
        #print(nodeCS.transform4x4)
        jointTransforms[nodeID] = nodeTransform


        for child in children:
            if hasattr(child,'id'):
                childID = child.id
                #if childID !='joint_Root':
                childID = childID.replace(self.anim_name + '_','')
                self.__updateGlobalJointTransforms(time,jointTransforms,childID,nodeID)
        return jointTransforms
    
    
    
      # update the rigid body approximation of joint-weighted meshes used by the RSS
    def updateRigidMeshes(self,time):
        
        transforms = self.__updateGlobalJointTransforms(time)
        mesh_dict = {}

        for nodeID in self.sceneTree:
            if nodeID in self.subMeshes:
                #print(nodeID)
                #file_name = self.save_path + nodeID + ".stl"
                file_name = self.mapped_stl_filenames[nodeID]
                mesh = self.subMeshes[nodeID]
                #doing all mesh transformations in main script so commented this out
                #mesh.transform(transforms[nodeID])
                temp_dict = {'file_name':file_name,'mesh':mesh,'transform':transforms[nodeID]}
                mesh_dict[nodeID] = temp_dict

        return mesh_dict
    
    
    
class AnimatedDAE_CMU:
    '''
    testing dae importt from https://mocap.cs.sfu.ca/
    
    There are some difficult names that are not consistent between sources
    of dae files. So for this I try my best to correct, but you might have to 
    change some hard coded values
    
    '''
    def __init__(self,filename,save_path = './'):
        
        self.all_possible_files = glob.glob('meshes/CMU_Meshes2/*.stl')
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        self.save_path = save_path
        self.global_rot = np.eye(3)
        self.global_pos = np.zeros(3)
        self.filename = filename
        
        if self.filename.endswith(".daez"):
            base_filename = os.path.basename(self.filename)
            base_filename = base_filename.replace(".daez",".dae")
            self.dae = collada.Collada(self.filename,zip_filename = base_filename)
        else:
            self.dae = collada.Collada(self.filename)
        #self.dae = collada.Collada(self.filename)


        if self.dae.assetInfo.upaxis == 'Y_UP':
            self.local2global = np.asarray(
              [[ 0, 0,+1, 0],\
              [+1, 0, 0, 0],\
              [ 0,+1, 0, 0],\
              [ 0, 0, 0,+1]])
            self.global_rot = self.local2global
        elif self.dae.assetInfo.upaxis == 'X_UP':
            self.local2global = np.asarray(
              [[ 0, +1,0, 0],\
              [0, 0, +1, 0],\
              [ +1,0, 0, 0],\
              [ 0, 0, 0,+1]])
            self.global_rot = self.local2global
        elif self.dae.assetInfo.upaxis == 'Z_UP':
            self.local2global = np.asarray(
              [[ +1, 0,+1, 0],\
              [0, +1, 0, 0],\
              [ 0,0, +1, 0],\
              [ 0, 0, 0,+1]])
            self.global_rot = self.local2global

            
        else: raise RuntimeError("unsupported up-axis in dae file")
        
        #the files I exported say they are Z_up but they are actually  Y_up
        self.global_rot = np.asarray(
              [[ 0, 0,+1],\
              [+1, 0, 0],\
              [ 0,+1, 0]])
        global2local = np.linalg.inv(self.local2global)
        # rescale to meters
        units2meters = self.dae.assetInfo.unitmeter
        self.rescaleTransform = np.ones((4,4))
        self.rescaleTransform[0:3,3] = units2meters*1
        color = [0.8,0.8,0.8]
        # build joint transform interpolations
        jointTransforms = {}


        # for anim in self.dae.animations:
        #     times = anim.sourceById[anim.name + "-Matrix-animation-input"]
        #     times = np.reshape(times,-1)
        #     self.times = times
        #     self.clipLength = max(self.clipLength,times[-1])
        #     transforms = anim.sourceById[anim.name + "-Matrix-animation-output-transform"]


        self.clipLength = -1
        child_names_by_id = {}
        scene = self.dae.scenes[0].nodes[0]
        


            
        for anim in self.dae.animations:
            name = list(anim.sourceById.keys())
            times = anim.sourceById[name[0]]
            times = np.reshape(times,-1)
            times = times-np.min(times) #zero shift
            self.times = times
            self.clipLength = max(self.clipLength,times[-1])
            self.animation_name = anim.name
        original_fps =1/np.average( np.diff(times))
        #get common string in animation names for splitting string later
        all_part_names = []
        for child in anim.children:
            all_part_names.append(child.id)

        string2 = all_part_names[0]
        all_matches = []
        for i in range(1, len(all_part_names)):
            string1 = string2
            string2 = all_part_names[i]
            match = SequenceMatcher(None, string1, string2).find_longest_match(0, len(string1), 0, len(string2))
            all_matches.append(string1[match.a: match.a + match.size])
        shortest_string = ''.zfill(255)
        for each in all_matches:
            if len(each)<len(shortest_string):
                shortest_string = each
        common_str = shortest_string

        #assuming only one animation, anim
        for child in anim.children:

            transforms = anim.sourceById[child.id + "-output"]
            transforms = np.reshape(transforms,(-1,4,4)) 
            transforms = transforms*self.rescaleTransform
            
            # HFSS requires euler angles for object orientations, the 3x3 rotational
            # matrix that is defined here, needs to eventually be converted to 
            # euler angles.
            # I found that if I use the rotation matrix, and create an iterpolation function
            # from it. When I convert to euler
            # angles, then unwrap the phase. I get interpolation errors.
            # What seems to work better, is convert rotational matrix to euler
            # then unwrap, the convert back to rotation matrix. then when I 
            # convert it back to euler and unwrap I don't get interpolation errors
            
            transforms2 = copy.deepcopy(transforms)
            rot_3x3 = transforms[:,0:3,0:3]
            euler_angs = rot_to_euler(rot_3x3,order='zyz',deg=False)

            phi = np.rad2deg(np.unwrap(euler_angs[:,0],period=np.pi*2))
            theta = np.rad2deg(np.unwrap(euler_angs[:,1],period=np.pi*2))
            psi = np.rad2deg(np.unwrap(euler_angs[:,2],period=np.pi*2))
            euler = np.vstack((phi,theta,psi)).T
            
            rot = euler_to_rot(euler)
            transforms2[:,0:3,0:3] = rot

            interpFunc = scipy.interpolate.interp1d(times,transforms2,axis=0,assume_sorted=True)
            #temporary renmaing until I figure out easier and more robust way to get names
            #sometimes the _pos_matrix exists, sometimes it doesn't
            childname =child.id.split(common_str)[-1].split('_pose_matrix')[0]#.replace('_',':')

            if childname=='transform':
                childname='sbui_Hips'


            jointTransforms[childname] = interpFunc

        def recurseJoints(
            sceneTree = None,
            nodeID = None):
              
            
            if sceneTree is None:
                nodeID = "root"
                #root_name = 
                nodeCS = CoordSys()
                nodeCS.set(self.local2global)
                nodeCS.update(time=0.,inGlobal=False)
                sceneTree = {}

                sceneTree[nodeID] = (nodeCS,scene.children)
            (nodeCS,children) = sceneTree[nodeID]
            for child in children:
                if hasattr(child, 'id'):
                    childID = child.id
                    childID = childID.split('_')[-1]
                    # if childID !='sbui_Hips':
                    #     childID = childID.replace(self.animation_name,'')
                    #print(childID)
                    childCS = CoordSys()
                    if childID in jointTransforms:
                        childCS.transforms = jointTransforms[childID]
    
                    else:
                        childTransform = child.matrix*self.rescaleTransform
                        childCS.set(childTransform)
                    childCS.update(time=0.,inGlobal=True)
                    sceneTree[childID] = (childCS,child.children)
                    recurseJoints(sceneTree,childID)
            return sceneTree
        self.sceneTree = recurseJoints()
        # split mesh into connected components
        self.subMeshes = {}

        self.all_possible_files
        meshes = []
        self.mapped_stl_filenames = {}
        scale_factor = 1
        up_legs = ['RightHip','LeftHip',]
        low_legs = ['RightKnee','LeftKnee']
        foot = ['LeftAnkle','RightAnkle']
        larms = ['LeftShoulder','LeftElbow']
        rarms = ['RightShoulder','RightElbow']
        rhand = ['rhand']
        lhand = ['lhand']
        rfoot = ['RightAnkle']
        foot = ['LeftAnkle','RightAnkle']
        rfoot = ['RightAnkle']
        lfoot = ['LeftAnkle']
        rtoe = ['RightToe']
        ltoe = ['LeftToe']
        for n, part in enumerate(self.sceneTree):
            if (part !='root' and  part !='Chest2'):
                #print('Part:' + part)
                # part_str = part.replace("_005_Walking001_0005_Walking001_","")
                # part_str = part_str.replace("_pose_matrix","")
                part_str = part.split("_")[-1]
                #print(part)
                for file in self.all_possible_files:
                    if part_str in file:
                        #print('Part String:' + part_str)
                        #print('File Name:' + file)
                        mesh = pv.read(file)
                        mesh.scale([scale_factor, scale_factor, scale_factor],inplace=True)
                        # if part == 'sbui_Spine1':
                        #     mesh.translate([0,-.05,0])
                        if part in low_legs:
                            mesh.rotate_x(180,inplace=True)
                            mesh.translate([0,0,-0.0625],inplace=True)
                        elif part in up_legs:
                            mesh.rotate_x(180,inplace=True)
                            mesh.rotate_y(180,inplace=True)
                            mesh.translate([0,0,-0.0625],inplace=True)
                        elif part in rarms:
                            mesh.rotate_y(-90,inplace=True)
                            mesh.rotate_x(90,inplace=True)
                            mesh.translate([0,0,-.03125],inplace=True)
                        elif part in larms:
                            mesh.rotate_z(-90,inplace=True)
                            mesh.rotate_y(-90,inplace=True)
                            mesh.rotate_z(180,inplace=True)
                            mesh.translate([0,0,-.03125],inplace=True)
                        elif part in lhand:
                            mesh.rotate_z(-90,inplace=True)
                            mesh.rotate_y(90,inplace=True)
                            mesh.rotate_z(90,inplace=True)
                        elif part in rhand:
                            mesh.rotate_z(-90,inplace=True)
                            mesh.rotate_y(-90,inplace=True)
                            mesh.rotate_z(-90,inplace=True)
                        elif part in lfoot:
                            mesh.rotate_z(-90,inplace=True)
                            mesh.rotate_y(90+45,inplace=True)
                            mesh.rotate_z(180,inplace=True)
                        elif part in rfoot:
                            mesh.rotate_z(-90,inplace=True)
                            mesh.rotate_y(180+45,inplace=True)
                            mesh.rotate_z(180,inplace=True)
                        elif part in rtoe:
                            mesh.rotate_z(-90,inplace=True)
                            mesh.rotate_y(-90,inplace=True)
                        elif part in ltoe:
                            mesh.rotate_z(-90,inplace=True)
                            mesh.rotate_y(-90,inplace=True)
                            mesh.rotate_z(180,inplace=True)
                        self.mapped_stl_filenames[part] = file
                        self.subMeshes[part]=mesh
                        break    
 

    
    def __updateGlobalJointTransforms(
        self,
        time,
        jointTransforms={},
        nodeID="root",
        parentID=None):
        
        global maxErrors
        (nodeCS,children) = self.sceneTree[nodeID]
        if parentID is None:
            #parentTransform=np.eye(4)
            nodeCS.pos = self.global_pos
            nodeCS.rot = self.global_rot
            nodeCS.update(time=time,inGlobal=False)
            parentTransform = nodeCS.transform4x4
        else:
            parentTransform = jointTransforms[parentID]


        #deal with loop
        time = time%self.clipLength
        
        nodeCS.update(time=time,inGlobal=False)
        nodeTransform = np.dot(parentTransform,nodeCS.transform4x4)
        #nodeTransform = nodeCS.transform4x4
        #print(nodeCS.transform4x4)
        # if nodeID=='LeftKnee':
        #     print(nodeCS.pos )
        #     print(nodeCS.rot )
        jointTransforms[nodeID] = nodeTransform


        for child in children:
            if hasattr(child,'id'):
                childID = child.id
                childID = childID.split('_')[-1]
                self.__updateGlobalJointTransforms(time,jointTransforms,childID,nodeID)
        return jointTransforms
    
    
    
      # update the rigid body approximation of joint-weighted meshes used by the RSS
    def updateRigidMeshes(self,time):
        
        transforms = self.__updateGlobalJointTransforms(time)
        mesh_dict = {}

        for nodeID in self.sceneTree:
            if nodeID in self.subMeshes:
                #print(nodeID)
                #file_name = self.save_path + nodeID + ".stl"
                file_name = self.mapped_stl_filenames[nodeID]
                mesh = self.subMeshes[nodeID]
                #doing all mesh transformations in main script so commented this out
                #mesh.transform(transforms[nodeID])
                temp_dict = {'file_name':file_name,'mesh':mesh,'transform':transforms[nodeID]}
                mesh_dict[nodeID] = temp_dict

        return mesh_dict