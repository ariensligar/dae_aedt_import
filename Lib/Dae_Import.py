import collada
import scipy
import scipy.interpolate
import numpy as np
import copy
import pyvista as pv
import os


class CoordSys:
  def __init__(self):

    self.time = 0
    self.rot = np.eye(3)
    self.pos = np.zeros(3)
    self.transforms = None

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
            self.clipLength = max(self.clipLength,times[-1])
            transforms = anim.sourceById[anim.name + "-Matrix-animation-output-transform"]
            transforms = np.reshape(transforms,(-1,4,4)) 
            transforms = transforms*self.rescaleTransform
            interpFunc = scipy.interpolate.interp1d(times,transforms,axis=0,assume_sorted=True)
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


                    # mesh.transform(transforms[nodeID])
                    # meshes.append(mesh)
                
        
        return mesh_dict