# -*- coding: utf-8 -*-
"""
Created on Mon Nov 22 11:46:09 2021

This script will import a DAE file into AEDT, along with setting up the rest of the
HFSS project to be ready to run for radar simulations

This is for testing how blender export a slightly different format of dae than
what is availble on mixamo.

There are a lot of variation depending on the sources of bvh/fbx file, so this is just an attempt
to convert a few known sources. There are a few hard coded values that may need to be changed
You can use this as an example but it may not work for every fbx

@author: asligar
"""

from pyaedt import Hfss
import pyvista as pv
import numpy as np
from Lib.Dae_Import import AnimatedDAE #this seems to work for animations downloaded from mixamo
from Lib.Dae_Import import AnimatedDAE2 #this works for certain fbx export
from Lib.Dae_Import import AnimatedDAE3 #this is an example of converting fbx files from https://mocap.cs.sfu.ca/
from Lib.Dae_Import import AnimatedDAE4 #this is an example of converting dae files from https://github.com/BandaiNamcoResearchInc/Bandai-Namco-Research-Motiondataset
from Lib.Dae_Import import rot_to_euler 
from Lib.Dae_Import import CoordSys
from Lib.AEDT_Utils import AEDTutils
from scipy.signal import savgol_filter
import copy
import glob

###############################################################################
#
# BEGIN USER INPUTS
#
###############################################################################

#full path to file name that we want to import
#IMPORTANT USE DAE  FILE

all_files = glob.glob('./example_dae/BandaiNamcoData/*.dae')
filename= np.random.choice(all_files)
print(filename)
#filename = './example_dae/BandaiNamcoData/dataset-1_walk-left_not-confident_001.dae'
#Define Framerate, ideally should be same as DAE file, but interpolation will allow any frame rate
fps=30

#if data is noisy, we can smooth it out
smoothing=False
#hands often have lots of small parts, we can remove them to speed up generation
remove_hands=True 
loop_animation=True
aedt_version = "2022.1"
create_AEDT_proj = False

###############################################################################
#
# END USER INPUTS
#
###############################################################################


def main(filename,fps,smoothing=False,remove_hands=True,aedt_version= "2021.2"):

    
    #import mesh
    ani = AnimatedDAE4(filename,save_path='./meshes/')
    
    #intial mesh dictionary at time 0
    meshes_dict = ani.updateRigidMeshes(time=0)
    
    #frame rate
    num_of_frames = int(ani.clipLength*fps)
    time_stamps = np.linspace(0,ani.clipLength,num=num_of_frames)
    
    #for testing, quick generation, remove hands which have lots of parts
    if remove_hands:
        temp = copy.copy(meshes_dict)
        for each in temp:
            if 'Hand' in each:
                del meshes_dict[each]
    
    if create_AEDT_proj:
        #instances of AEDT
        #intialize, gets project and design name, maybe a better way to do this, I'll revist later
        aedt = AEDTutils(project_name='dae_import_example',design_name='design1',version=aedt_version)
        
        #instance of HFSS
        with Hfss(projectname=aedt.project_name,
                      designname=aedt.design_name,
                      non_graphical=False, 
                      new_desktop_session=False,
                      specified_version=aedt_version,
                      solution_type='SBR+') as aedtapp:
            
            aedt.aedtapp = aedtapp
            
    
            aedt.setup_design(time_stamps)
            #add material property assigned to CAD
            aedt.add_material('human_avg',5,0.01,0)
            
            #base CS where ped will be placed
            base_cs = aedt.create_cs('ped_ref_cs',pos=[0,0,0],euler=[90,90,90]) #rotating reference to Z is up (dae files form mixamo are y+ up)
            pos_dict = {}
            euler_dict = {}
            
            for node_id in meshes_dict:
                print(f'Importing {node_id}')
                file_name =meshes_dict[node_id]['file_name']
                
                pos_x=[]
                pos_y=[]
                pos_z=[]
                phi = []
                theta = []
                psi = []
                
                node_id_str = node_id.replace('-','_').split('_')
                node_id_str = '_'.join(node_id_str[-4:-1])
                node_id_str = node_id
                x_ds_name = f'{node_id_str}_pos_x_ds'
                y_ds_name = f'{node_id_str}_pos_y_ds'
                z_ds_name = f'{node_id_str}_pos_z_ds'
                phi_ds_name = f'{node_id_str}_phi_ds'
                theta_ds_name = f'{node_id_str}_theta_ds'
                psi_ds_name = f'{node_id_str}_psi_ds'
                
                pos_ds_names = {'x':x_ds_name,'y':y_ds_name,'z':z_ds_name}
                euler_ds_names = {'phi':phi_ds_name,'theta':theta_ds_name,'psi':psi_ds_name}
                
                #get all transforms for all time steps
                for time in time_stamps:
                    #get new transforms for time step
                    update_transform = ani.updateRigidMeshes(time=time)
                    transform =update_transform[node_id]['transform']
                    
                    #coordinate system instance that allows 4x4transformmatrix to be converted
                    # 3x3 rotation matrix and 3x1 positions (cs.rot and cs.pos)
                    cs = CoordSys()
                    cs.set(transform)
            
                    #get euler angles from rotation matrix
                    euler_angles = rot_to_euler(cs.rot,order='zyz')
                    pos = cs.pos
            
                    pos_x.append(pos[0])
                    pos_y.append(pos[1])
                    pos_z.append(pos[2])
            
                    #why do I need these addition rotations?!
                    #I am not sure why, but they work. 
                    phi.append(euler_angles[2]+180)
                    theta.append(-euler_angles[1])
                    psi.append(euler_angles[0]-180)
                
                if smoothing:
                    pos_x = savgol_filter(pos_x,5,polyorder=3)
                    pos_y = savgol_filter(pos_y,5,polyorder=3)
                    pos_z = savgol_filter(pos_z,5,polyorder=3)
                
                #unwrap otherwise you can end up with interpolation errors between points
                #when phase flips
                phi = np.rad2deg(np.unwrap(np.deg2rad(phi),period=np.pi*2))
                theta = np.rad2deg(np.unwrap(np.deg2rad(theta),period=np.pi*2))
                psi = np.rad2deg(np.unwrap(np.deg2rad(psi),period=np.pi*2))
                if smoothing:
                    phi = savgol_filter(phi,5,polyorder=3)
                    theta = savgol_filter(theta,5,polyorder=3)
                    psi = savgol_filter(psi,5,polyorder=3)
                
                pos_dict[node_id] = np.array([pos_x,pos_y,pos_z]).T
                euler_dict[node_id] = np.array([phi,theta,psi]).T
            
                #create datasets for each body part
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
                
                #create dataset for each body part
                cs_name = aedt.create_cs_dataset(node_id_str+'_cs',
                                                  pos_ds_names=pos_ds_names,
                                                  euler_ds_names=euler_ds_names,
                                                  reference_cs=base_cs,
                                                  loop_animation=loop_animation)
            
                #using material propety human_avg for assignment
                imported_names = aedt.import_stl(file_name, cs_name=cs_name)
                aedt.assign_material(imported_names,'pec')
                bc_name= aedt.assign_boundary(imported_names,'human_avg',bc_name=node_id_str+ "_bc")
                
                
                legs = ['sbui_LeftLeg','sbui_RightLeg','sbui_LeftUpLeg','sbui_RightUpLeg']
                foot = ['sbui_LeftFoot','sbui_RightFoot']
                larms = ['sbui_LeftArm','sbui_LeftForeArm']
                rarms = ['sbui_RightArm','sbui_RightForeArm']
                
                # legs = ['UpperLeg_L','LowerLeg_L','LowerLeg_R','UpperLeg_R']
                # foot = ['Foot_L','Foot_R']
                # larms = ['LowerArm_L','UpperArm_L']
                # rarms = ['UpperArm_R','LowerArm_R']
                # if part == 'sbui_Spine1':
                #     mesh.translate([0,-.05,0])
                #need to add some rotation
                print(imported_names)
                if node_id_str in legs:
                    #
                    aedt.rotate(imported_names,single_rot=180,axis='X',reference_cs=cs_name)
                if node_id_str in foot:
                    aedt.rotate(imported_names,single_rot=-90,axis='X',reference_cs=cs_name)
                    aedt.rotate(imported_names,single_rot=180,axis='Y',reference_cs=cs_name)
                if node_id_str in rarms:
                    aedt.rotate(imported_names,single_rot=-90,axis='Y',reference_cs=cs_name)
                    aedt.rotate(imported_names,single_rot=90,axis='X',reference_cs=cs_name)
                if node_id_str in larms:
                    aedt.rotate(imported_names,single_rot=180,axis='X',reference_cs=cs_name)
                    aedt.rotate(imported_names,single_rot=-90,axis='Y',reference_cs=cs_name)
                    aedt.rotate(imported_names,single_rot=90,axis='X',reference_cs=cs_name)
                    aedt.rotate(imported_names,single_rot=180,axis='Z',reference_cs=cs_name)
                aedt.convert_to_3d_comp(node_id_str,cs_name,parts = imported_names,boundary_conditions=bc_name)
                
                
            #create CS for radar location, create simple parametric tx/rx antenna
            radar_cs = aedt.create_cs('radar_cs',pos=[10,0,0],euler=[180,0,0])
            aedt.insert_parametric_antenna('tx','30deg','80deg','Vertical',cs=radar_cs)
            aedt.insert_parametric_antenna('rx','30deg','80deg','Vertical',cs=radar_cs)
            aedt.set_tx_rx()
            
            #add simulation setup with the following parameters
            simulation_params={}
            simulation_params['sol_freq'] = 76.5
            simulation_params['range_res'] = 0.1
            simulation_params['range_period']=20
            simulation_params['vel_res']=0.1
            simulation_params['vel_min']=-10
            simulation_params['vel_max']=10
            simulation_params['ray_density']=0.2
            simulation_params['bounces']=2
            setup_name=aedt.insert_setup(simulation_params,setup_name = "Setup1")
            
            #create parameteric sweep, last time step is not valid so removing it.
            para_sweep_name = aedt.insert_parametric_sweep(0,time_stamps[-2],1/fps,setup_name)
            #aedt.release_desktop()          
    
    ###############################################################################
    #
    # VISUALIZATION USING PYVISTA (only needed for testing)
    #
    ###############################################################################
    plotter = pv.Plotter()
    #add intial meshes
    for node_id in meshes_dict:
        plotter.add_mesh(meshes_dict[node_id]['mesh'],name=node_id)
        
    #create interactive plotting for time animation
    #creates slider bar with time to adjust animation time selection
    def time_select(value=0):
        time = value #slider doesn't allow discrete values
        updated_meshes_dict = ani.updateRigidMeshes(time=time)
        for nodeid in meshes_dict:
            #plotter.clear
            mesh = copy.deepcopy(updated_meshes_dict[nodeid]['mesh'])
            #print(updated_meshes_dict[nodeid]['transform'][0:3,3])
            mesh.transform(updated_meshes_dict[nodeid]['transform'])
            cs = CoordSys()
            cs.set(updated_meshes_dict[nodeid]['transform'])
            
            #testing, just doing indivudal rotation to help troublshoot aedt rotations
            #euler = rot_to_euler(cs.rot,order='zyz')
            # mesh.rotate_z(euler[0])
            # mesh.rotate_y(euler[1])
            # mesh.rotate_z(euler[2])
            # mesh.translate(cs.pos)
            mesh.rotate_x(90) #rotate view so Z is up
            plotter.add_mesh(mesh,name=nodeid)
        return
    time_slider = plotter.add_slider_widget(time_select, [0, ani.clipLength], 
                                            title='Time Select',
                                            value=0,
                                            event_type='always')
    
    
    #for testing, just allowing point selection to determine if in correct location
    plotter.enable_point_picking( show_message=True, 
                           color='pink', point_size=10, 
                           use_mesh=True, show_point=True)
    plotter.add_axes(line_width=5)
    plotter.show()
    
    

    
    #testing,  
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


if __name__ == "__main__":
    main(filename,fps,smoothing=False,remove_hands=True,aedt_version= aedt_version)