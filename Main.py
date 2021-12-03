# -*- coding: utf-8 -*-
"""
Created on Mon Nov 22 11:46:09 2021

@author: asligar
"""

import pyvista as pv
import numpy as np
from Lib.Dae_Import import AnimatedDAE
from Lib.Dae_Import import CoordSys
from Lib.AEDT_Utils import AEDTutils
from scipy.spatial.transform import Rotation
import copy
filename = 'C:\\Users\\asligar\\OneDrive - ANSYS, Inc\\Desktop\\delete\\Falling.dae'
#filename = 'C:\\Users\\asligar\\OneDrive - ANSYS, Inc\\Desktop\\delete\\Walking.dae'
 



def rot_to_euler(rot, order='zyz',deg=True):
    rot = Rotation.from_matrix(rot)
    euler_angles = rot.as_euler(order,degrees=deg)
    #euler_angles = rot.as_rotvec(degrees=deg)
    return euler_angles, rot

def rotmat_to_yaw_pitch_roll(rot,deg=True):
    '''
    

    Parameters
    ----------
    rot : TYPE
        DESCRIPTION.
    deg : boolean, optional
        DESCRIPTION. if output will be in degree or radians.

    Returns
    -------
    yaw : TYPE
        DESCRIPTION.
    pitch : TYPE
        DESCRIPTION.
    roll : TYPE
        DESCRIPTION.

    '''
    if deg==True:
        convert = 180/np.pi
    else:
        convert=1
    yaw = np.arctan2(rot[1,0], rot[0,0])*convert
    pitch = np.arccos(rot[2,2])*convert
    roll = np.arctan2(rot[2,1], rot[2,2])*convert
    

    
    return [yaw,pitch,roll]




ani = AnimatedDAE(filename,save_path='./meshes/')
meshes_dict = ani.updateRigidMeshes(time=0)

#define requested frame rate
fps=100
num_of_frames = int(ani.clipLength*fps)
time_stamps = np.linspace(0,ani.clipLength,num=num_of_frames)

plotter = pv.Plotter()


#add intial meshes
for node_id in meshes_dict:
    plotter.add_mesh(meshes_dict[node_id]['mesh'],name=node_id)
    
#create interactive plotting for time animation
def time_select(value=0):
    time = value #slider doesn't allow discrete values
    updated_meshes_dict = ani.updateRigidMeshes(time=time)
    for nodeid in meshes_dict:
        plotter.clear
        mesh = copy.deepcopy(updated_meshes_dict[nodeid]['mesh'])
        mesh.transform(updated_meshes_dict[nodeid]['transform'])
        cs = CoordSys()
        cs.set(updated_meshes_dict[nodeid]['transform'])
        euler,_ = rot_to_euler(cs.rot,order='zyz')

        #testing, just doing indivudal rotation to help troublshoot aedt rotations
        # mesh.rotate_z(euler[0])
        # mesh.rotate_y(euler[1])
        # mesh.rotate_z(euler[2])
        # mesh.translate(cs.pos)
        plotter.add_mesh(mesh,name=nodeid)
    return
time_slider = plotter.add_slider_widget(time_select, [0, ani.clipLength], title='Time Select',value=0)



plotter.enable_point_picking( show_message=True, 
                       color='pink', point_size=10, 
                       use_mesh=True, show_point=True)

plotter.show()

meshes_dict = ani.updateRigidMeshes(time=0)
# test = meshes_dict['mixamorig_LeftArm']['transform']
# cs = CoordSys()
# cs.set(test)
# euler_angles = rot_to_euler(cs.rot,order='zyx')

#for testing, quick generation, remove hands which have lots of parts
temp = copy.copy(meshes_dict)
for each in temp:
    key_name = str(each)
    if 'Hand' in each:
        del meshes_dict[each]
        
aedt = AEDTutils()
aedt.add_material('human_avg',5,0.01,0)

base_cs = aedt.create_cs('ped_ref_cs',pos=[0,0,0],euler=[90,90,90]) #rotating reference to Z is up (dae files form mixamo are y+ up)
pos_dict = {}
euler_dict = {}
for node_id in meshes_dict:
    print(node_id)
    mesh =meshes_dict[node_id]['mesh']

    file_name =meshes_dict[node_id]['file_name']
    
    pos_x=[]
    pos_y=[]
    pos_z=[]
    phi = []
    theta = []
    psi = []
    
    x_ds_name = f'{node_id}_pos_x_ds'
    y_ds_name = f'{node_id}_pos_y_ds'
    z_ds_name = f'{node_id}_pos_z_ds'
    phi_ds_name = f'{node_id}_phi_ds'
    theta_ds_name = f'{node_id}_theta_ds'
    psi_ds_name = f'{node_id}_psi_ds'
    
    pos_ds_names = {'x':x_ds_name,'y':y_ds_name,'z':z_ds_name}
    euler_ds_names = {'phi':phi_ds_name,'theta':theta_ds_name,'psi':psi_ds_name}
    

    #get all transforms for all time steps
    for time in time_stamps:
        update_transform = ani.updateRigidMeshes(time=time)
        transform =update_transform[node_id]['transform']
        cs = CoordSys()
        cs.set(transform)
        euler_angles,rot_func = rot_to_euler(cs.rot,order='zyz')
        pos = cs.pos
        #pos = rot_func.apply(cs.pos)
        #euler_angles = rotmat_to_yaw_pitch_roll(cs.rot)
        pos_x.append(pos[0])
        pos_y.append(pos[1])
        pos_z.append(pos[2])

        #why do I need these addition rotations?!
        #I am not sure why, but they work. 
        phi.append(euler_angles[2]+180)
        theta.append(-euler_angles[1])
        psi.append(euler_angles[0]-180)
        
    pos_dict[node_id] = np.array([pos_x,pos_y,pos_z]).T
    euler_dict[node_id] = np.array([phi,theta,psi]).T
    pos_x_ds = zip(time_stamps,pos_x)
    pos_y_ds = zip(time_stamps,pos_y)
    pos_z_ds = zip(time_stamps,pos_z)
    phi_ds = zip(time_stamps,phi)
    theta_ds = zip(time_stamps,theta)
    psi_ds = zip(time_stamps,psi)
    
    aedt.add_dataset(x_ds_name,pos_x_ds)    
    aedt.add_dataset(y_ds_name,pos_y_ds)
    aedt.add_dataset(z_ds_name,pos_z_ds)    
    aedt.add_dataset(phi_ds_name,phi_ds)
    aedt.add_dataset(theta_ds_name,theta_ds)
    aedt.add_dataset(psi_ds_name,psi_ds)
    
    cs_name = aedt.create_cs_dataset(node_id+'_cs',pos_ds_names=pos_ds_names,euler_ds_names=euler_ds_names,reference_cs=base_cs)

    imported_names = aedt.import_stl(file_name, cs_name=cs_name)
    aedt.assign_material(imported_names,'human_avg')
    aedt.assign_boundary(imported_names,'human_avg',bc_name=node_id+ "_bc")
    aedt.convert_to_3d_comp(node_id,cs_name)
    
    # aedt.rotate(node_id,rot_ds_name=phi_ds_name,axis='Z',reference_cs='Global')
    # aedt.rotate(node_id,rot_ds_name=theta_ds_name,axis='X')
    # aedt.rotate(node_id,rot_ds_name=psi_ds_name,axis='Z')
    # aedt.move(node_id,pos_ds_names,reference_cs='Global')
    
radar_cs = aedt.create_cs('radar_cs',pos=[10,0,0])
aedt.insert_parametric_antenna('tx','30deg','80deg','Vertical',cs=radar_cs)
aedt.insert_parametric_antenna('rx','30deg','80deg','Vertical',cs=radar_cs)
aedt.set_tx_rx()

#testing, why 
# plotter = pv.Plotter()
# #add intial meshes
# idx_to_plot = 0
# for nodeid in meshes_dict:
#     print(nodeid)
#     plotter.clear
#     mesh = copy.deepcopy(meshes_dict[nodeid]['mesh'])
#     #mesh.transform(meshes_dict[nodeid]['transform'])

#     euler = euler_dict[nodeid][idx_to_plot]
#     pos = pos_dict[nodeid][idx_to_plot]
#     #mesh.translate(cs.pos)
#     mesh.rotate_z(euler[0])
#     mesh.rotate_x(euler[1])
#     mesh.rotate_z(euler[2])
#     mesh.translate(pos)
#     print(pos)
#     plotter.add_mesh(mesh,name=nodeid)
# plotter.show()
