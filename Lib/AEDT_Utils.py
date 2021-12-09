# -*- coding: utf-8 -*-
"""
Created on Mon Nov 22 11:46:09 2021

interaction between main script and aedt is defined here, assumes
that the desktop is open, uses current active project

@author: asligar
"""

from pyaedt import Hfss
from pyaedt import Desktop
import os
import uuid

class AEDTutils:
    def __init__(self,project_name='project1',design_name='design1',version ="2021.2"):
        

        self.desktop = Desktop(specified_version=version,new_desktop_session =False,close_on_exit =False)
        projects = self.desktop.project_list()
        if project_name in projects:
            increment=1
            orig_proj_name = project_name
            while  project_name in projects:
                project_name = orig_proj_name+'_' + str(increment)
                increment+=1
            self.desktop.odesktop.SetActiveProject(project_name)
        
            designs = self.desktop.design_list()
            orig_design_name=design_name
            increment=1
            while  design_name in designs:
                design_name = orig_design_name+str(increment)
                increment+=1
            
        self.aedtapp = Hfss(projectname=project_name,designname=design_name,specified_version=version,solution_type='SBR+')
        
        self.oDesign = self.aedtapp.odesign
        oEditor = self.oDesign.SetActiveEditor("3D Modeler")
        oEditor.SetModelUnits(["NAME:Units Parameter","Units:=", "meter","Rescale:=", False])
        
        self.time_var_name = "time_var"
        self.time = 0
        self.add_or_edit_variable(self.time_var_name,str(self.time)+'s')
                

    def release_desktop(self):
        self.desktop.release_desktop(close_projects=False, close_on_exit=False)

    def diff(self,li1, li2): 
        """
        used to return difference between two lists
        commonly used for when HFSS doesn't return the name of objects, for example
        when an stl file is imported, this function can be used to compare list
        of objects before and after import to return the list of imported objects

        returns: difference between lists
        """
        li_dif = [i for i in li1 + li2 if i not in li1 or i not in li2] 
        return li_dif 
    
    def add_material(self,mat_name,er,tand,cond):
        """
        adds a material to HFSS with the properties of permitivity and dielectric
        loss tanget only. If the material already exists, it will update it with
        the new properties
        """
        oDefinitionManager = self.aedtapp.oproject.GetDefinitionManager()
        existing_materials = oDefinitionManager.GetInUseProjectMaterialNames()
        if mat_name in existing_materials:
            oDefinitionManager.EditMaterial(mat_name, 
            [
                "NAME:"+mat_name,
                "CoordinateSystemType:=", "Cartesian",
                "BulkOrSurfaceType:="    , 1,
                [
                    "NAME:PhysicsTypes",
                    "set:="            , ["Electromagnetic"]
                ],
                "permittivity:="    , str(er),
                "dielectric_loss_tangent:=", str(tand),
                "bulk_conductivity:=", str(cond)
            ])
        else:
            oDefinitionManager.AddMaterial(
                [
                    "NAME:"+mat_name,
                    "CoordinateSystemType:=", "Cartesian",
                    "BulkOrSurfaceType:="    , 1,
                    [
                        "NAME:PhysicsTypes",
                        "set:="            , ["Electromagnetic"]
                    ],
                    "permittivity:="    , str(er),
                    "dielectric_loss_tangent:=", str(tand),
                    "bulk_conductivity:=", str(cond)
                ])

    def assign_material(self,obj_names,material):
        oEditor = self.aedtapp.odesign.SetActiveEditor("3D Modeler")
        if isinstance(obj_names, list):
            for each in obj_names:
                oEditor.ChangeProperty(
                    [
                        "NAME:AllTabs",
                        [
                            "NAME:Geometry3DAttributeTab",
                            [
                                "NAME:PropServers", 
                                each
                            ],
                            [
                                "NAME:ChangedProps",
                                [
                                    "NAME:Material",
                                    "Value:="        , "\"" + material + "\""
                                ]
                            ]
                        ]
                    ])
        else:
            oEditor.ChangeProperty(
                [
                    "NAME:AllTabs",
                    [
                        "NAME:Geometry3DAttributeTab",
                        [
                            "NAME:PropServers", 
                            obj_names
                        ],
                        [
                            "NAME:ChangedProps",
                            [
                                "NAME:Material",
                                "Value:="        , "\"" + material + "\""
                            ]
                        ]
                    ]
                ])
        
    def assign_boundary(self,objects,material,bc_name="layered_bc1"):
        """
        assigns infinintly thick layered impedance boundary or pec boundary, 1 sided only type being used
        current, eventually expand to include 2 sided boundaries
        """
        oModule = self.aedtapp.odesign.GetModule("BoundarySetup")

        if material=="pec":
            existing_boundary_name = oModule.GetBoundariesOfType('Perfect E')
            n=1
            original_name = bc_name
            while bc_name in existing_boundary_name:
                bc_name = original_name + str(n)
                n+=1
            oModule.AssignPerfectE(
                [
                    "NAME:"+bc_name,
                    "Objects:="        , objects
                ])    
        else:
            existing_boundary_name = oModule.GetBoundariesOfType('Layered Impedance')
            n=1
            original_name = bc_name
            while bc_name in existing_boundary_name:
                bc_name = original_name + str(n)
                n+=1
            oModule.AssignLayeredImp(
                [
                    "NAME:"+bc_name,
                    "Objects:="        , objects,
                    "Frequency:="        , "0GHz",
                    "Roughness:="        , "0um",
                    "IsTwoSided:="        , False,
                    [
                        "NAME:Layers",
                        [
                            "NAME:Layer1",
                            "LayerType:="        , "Infinite",
                            "Thickness:="        , "1um",
                            "Material:="        , material
                        ]
                    ]
                ])
        return bc_name

    def set_tx_rx(self,tx_wildcard="tx",rx_wildcard="rx"):
        """
        Set Excitation to be used in the simulation to only excite the Tx antennas
        that way we don't need to run a simulation for every Tx-Rx and Rx-Tx pair
        only Tx to Rx.
        """
        oModule = self.aedtapp.odesign.GetModule("BoundarySetup")
        
        all_excitations = oModule.GetExcitationsOfType("Antenna Port")

        rx_str = ""
        tx_str = ""
        rx_list = []
        tx_list = []
        #make lists of all rx and tx antennas
        for each in all_excitations:
            #any port that has a "rx" in it is assigned rx, otherwise assign to be tx
            if rx_wildcard in each.lower():
                rx_list.append(str(each))
                rx_str = rx_str + str(each) + ","
            else:
                tx_list.append(str(each))

        #remove trailing comma
        rx_str = rx_str[:-1]    
        all_tx_rx_lists = ["NAME:SBRTxRxSettings"]

        #generate a the list of strings so all Tx and all Rx antennas pairs will be created
        for n, tx in enumerate(tx_list):
            all_tx_rx_lists.append(
                [
                    "NAME:Tx/Rx List "+str(n),
                    "Tx Antenna:="        , tx,
                    "Rx Antennas:="        , rx_str
                ])

        #set all antenna ports with the name 'tx' in them to be transmitters and similiar for 'rx'
        oModule.SetSBRTxRxSettings(    all_tx_rx_lists)
    def insert_parametric_antenna(self,name,beamwidth_el,beamwidth_az,polarization,cs="Global"):
        """
        this creates a single parmetric antenna component and inserts into coordinate system

        returns name of inserted antenna
        """
        uid = uuid.uuid4()

        oEditor = self.aedtapp.odesign.SetActiveEditor("3D Modeler")

        oEditor.InsertNativeComponent(
            [
                "NAME:InsertNativeComponentData",
                "TargetCS:="        , cs,
                "SubmodelDefinitionName:=", name,
                [
                    "NAME:ComponentPriorityLists"
                ],
                "NextUniqueID:="    , 0,
                "MoveBackwards:="    , False,
                "DatasetType:="        , "ComponentDatasetType",
                [
                    "NAME:DatasetDefinitions"
                ],
                [
                    "NAME:BasicComponentInfo",
                    "ComponentName:="    , name,
                    "Company:="        , "",
                    "Company URL:="        , "",
                    "Model Number:="    , "",
                    "Help URL:="        , "",
                    "Version:="        , "1.0",
                    "Notes:="        , "",
                    "IconType:="        , "Parametric Beam"
                ],
                [
                    "NAME:GeometryDefinitionParameters",
                    [
                        "NAME:VariableOrders"
                    ]
                ],
                [
                    "NAME:DesignDefinitionParameters",
                    [
                        "NAME:VariableOrders"
                    ]
                ],
                [
                    "NAME:MaterialDefinitionParameters",
                    [
                        "NAME:VariableOrders"
                    ]
                ],
                "MapInstanceParameters:=", "DesignVariable",
                "UniqueDefinitionIdentifier:=", str(uid),
                "OriginFilePath:="    , "",
                "IsLocal:="        , False,
                "ChecksumString:="    , "",
                "ChecksumHistory:="    , [],
                "VersionHistory:="    , [],
                [
                    "NAME:NativeComponentDefinitionProvider",
                    "Type:="        , "Parametric Beam",
                    "Unit:="        , "meter",
                    "Is Parametric Array:="    , False,
                    "Size:="        , "0.1meter",
                    "MatchedPortImpedance:=", "50ohm",
                    "Polarization:="    , polarization,
                    "Representation:="    , "Far Field",
                    "Vertical BeamWidth:="    , beamwidth_el,
                    "Horizontal BeamWidth:=", beamwidth_az
                ],
                [
                    "NAME:InstanceParameters",
                    "GeometryParameters:="    , "",
                    "MaterialParameters:="    , "",
                    "DesignParameters:="    , ""
                ]
            ])
        return name
    def import_stl(self,file_name,cs_name='Global'):
        self.aedtapp.modeler.set_working_coordinate_system(cs_name)
        full_stl_path = os.path.abspath(file_name)
        oEditor = self.oDesign.SetActiveEditor("3D Modeler")
        all_objects_before_import = oEditor.GetMatchedObjectName("*")
        oEditor.Import(
            [
                "NAME:NativeBodyParameters",
                "HealOption:="        , 0,
                "Options:="        , "-1",
                "FileType:="        , "UnRecognized",
                "MaxStitchTol:="    , -1,
                "ImportFreeSurfaces:="    , False,
                "GroupByAssembly:="    , False,
                "CreateGroup:="        , True,
                "STLFileUnit:="        , "meter",
                "MergeFacesAngle:="    , 0.02,
                "HealSTL:="        , False,
                "ReduceSTL:="        , False,
                "ReduceMaxError:="    , 0,
                "ReducePercentage:="    , 100,
                "PointCoincidenceTol:="    , 1E-06,
                "CreateLightweightPart:=", True,
                "ImportMaterialNames:="    , False,
                "SeparateDisjointLumps:=", False,
                "SourceFile:="        , full_stl_path
            ])
        all_objects_after_import = oEditor.GetMatchedObjectName("*")
        name_of_objects_imported = self.diff(all_objects_before_import,all_objects_after_import )
        return name_of_objects_imported

        
    def convert_to_3d_comp(self,name,cs_name):
        oEditor = self.oDesign.SetActiveEditor("3D Modeler")
        oEditor.ReplaceWith3DComponent(
            [
                "NAME:ReplaceData",
                "ComponentName:="    , name,
                "Company:="        , "",
                "Company URL:="        , "",
                "Model Number:="    , "",
                "Help URL:="        , "",
                "Version:="        , "1.0",
                "Notes:="        , "",
                "IconType:="        , "",
                "Owner:="        , "Arien Sligar",
                "Email:="        , "",
                "Date:="        , "11:20:05 AM  Dec 03, 2021",
                "HasLabel:="        , False,
                "IncludedParts:="    , [name],
                "HiddenParts:="        , [],
                "IncludedCS:="        , [cs_name],
                "ReferenceCS:="        , cs_name,
                "IncludedParameters:="    , [],
                "IncludedDependentParameters:=", [],
                "ParameterDescription:=", []
            ], 
            [
                "NAME:DesignData"
            ], 
            [
                "NAME:ImageFile",
                "ImageFile:="        , ""
            ])
    
    def add_or_edit_variable(self,name,value):
        self.aedtapp[name]=value
    
    def add_dataset(self,name,data):
        '''
        Adds a data set in HFSS. If a data set already exists, it will first
        delete it, then add it
        Parameters
        ----------
        name : str
            name of data set to be created or edited.
        data : 2d list
            values used in data set.
        Returns
        -------
        None.
        '''
        
        oDesign = self.aedtapp.odesign    
        temp_data = ["NAME:Coordinates"]
        for each in data:
            temp_data.append(["NAME:Coordinate","X:=", float(each[0]),"Y:=",
                float(each[1])])        
        ds = ["NAME:"+ name,temp_data]
    
        if oDesign.HasDataset(name) == True:
            oDesign.EditDataset(name,ds)
        else:
            oDesign.AddDataset(ds)
    
    def move(self,object_name,pos_ds_names,reference_cs='Global'):
        if pos_ds_names:
            if 'x' in pos_ds_names.keys():
                x=f"pwl({pos_ds_names['x']},{self.time_var_name})"
            else:
                x='0'
            if 'y' in pos_ds_names.keys():
                y=f"pwl({pos_ds_names['y']},{self.time_var_name})"
            else:
                y='0'
            if 'z' in pos_ds_names.keys():
               z=f"pwl({pos_ds_names['z']},{self.time_var_name})"
            else:
               z='0'
        oEditor = self.oDesign.SetActiveEditor("3D Modeler")
        self.aedtapp.modeler.set_working_coordinate_system(reference_cs)
        oEditor.Move(
            [
                "NAME:Selections",
                "Selections:="        , object_name,
                "NewPartsModelFlag:="    , "Model"
            ], 
            [
                "NAME:TranslateParameters",
                "TranslateVectorX:="    , x,
                "TranslateVectorY:="    , y,
                "TranslateVectorZ:="    , z
            ])
        
    def rotate(self,object_name,rot_ds_name,axis='X',reference_cs='Global'):
        rotate=f"pwl({rot_ds_name},{self.time_var_name})*1deg"

        oEditor = self.oDesign.SetActiveEditor("3D Modeler")
        self.aedtapp.modeler.set_working_coordinate_system(reference_cs)
        oEditor.Rotate(
        [
            "NAME:Selections",
            "Selections:="        , object_name,
            "NewPartsModelFlag:="    , "Model"
        ], 
        [
            "NAME:RotateParameters",
            "RotateAxis:="        , axis,
            "RotateAngle:="        , rotate
        ])
    def create_cs(self,cs_name,pos=[0,0,0],euler=[0,0,0],reference_cs='Global',order='ZYZ'):
       oEditor = self.oDesign.SetActiveEditor("3D Modeler")
       
       self.aedtapp.modeler.set_working_coordinate_system(reference_cs)
       
       exisiting_cs = oEditor.GetCoordinateSystems()
       
       orig_name = cs_name
       incrment = 1
       while cs_name in exisiting_cs:
           cs_name = orig_name + '_'+str(incrment)
           incrment+=1
       else: #creates new CS 

           oEditor.CreateRelativeCS(
               [
                   "NAME:RelativeCSParameters",
                   "Mode:="        , "Euler Angle "+ order,
                   "OriginX:="        , str(pos[0]),
                   "OriginY:="        , str(pos[1]),
                   "OriginZ:="        , str(pos[2]),
                   "Psi:="            , str(euler[0]) + 'deg',
                   "Theta:="        , str(euler[1]) + 'deg',
                   "Phi:="            , str(euler[2]) + 'deg'
               ], 
               [
                   "NAME:Attributes",
                   "Name:="        , cs_name
               ])


       self.aedtapp.modeler.set_working_coordinate_system(cs_name)
       return cs_name
   
    
    def create_cs_dataset(self,cs_name,pos_ds_names=None,euler_ds_names=None,reference_cs='Global',order='ZYZ'):
        oEditor = self.oDesign.SetActiveEditor("3D Modeler")
        
        self.aedtapp.modeler.set_working_coordinate_system(reference_cs)
        
        exisiting_cs = oEditor.GetCoordinateSystems()
        
        if pos_ds_names:
            if 'x' in pos_ds_names.keys():
                x=f"pwl({pos_ds_names['x']},{self.time_var_name})"
            else:
                x='0'
            if 'y' in pos_ds_names.keys():
                y=f"pwl({pos_ds_names['y']},{self.time_var_name})"
            else:
                y='0'
            if 'z' in pos_ds_names.keys():
               z=f"pwl({pos_ds_names['z']},{self.time_var_name})"
            else:
               z='0'
        else:
            x='0'
            y='0'
            z='0'
        
        if euler_ds_names:
            if 'phi' in euler_ds_names.keys():
                phi=f"pwl({euler_ds_names['phi']},{self.time_var_name})*1deg"
            else:
                phi='0deg'
            if 'theta' in euler_ds_names.keys():
                theta = f"pwl({euler_ds_names['theta']},{self.time_var_name})*1deg"
            else:
                theta='0deg'
            if 'psi' in euler_ds_names.keys():
                psi = f"pwl({euler_ds_names['psi']},{self.time_var_name})*1deg"
            else:
                psi='0deg'
        else:
            phi='0deg'
            theta='0deg'
            psi='0deg'
        orig_name = cs_name
        incrment = 1
        while cs_name in exisiting_cs:
            cs_name = orig_name + '_'+str(incrment)
            incrment+=1
        else: #creates new CS 

            oEditor.CreateRelativeCS(
                [
                    "NAME:RelativeCSParameters",
                    "Mode:="        , "Euler Angle "+ order,
                    "OriginX:="        , x,
                    "OriginY:="        , y,
                    "OriginZ:="        , z,
                    "Psi:="            , psi,
                    "Theta:="        , theta,
                    "Phi:="            , phi
                ], 
                [
                    "NAME:Attributes",
                    "Name:="        , cs_name
                ])

        self.aedtapp.modeler.set_working_coordinate_system(cs_name)
        return cs_name


    def insert_setup(self,simulation_params,setup_name = "Setup1"):
        """
        insert a solution setup, these settings can be modified as needed
        """
        oModule = self.oDesign.GetModule("AnalysisSetup")


        oModule.InsertSetup("HfssDriven", 
            [
                "NAME:"+ setup_name,
                "IsEnabled:="        , True,
                [
                    "NAME:MeshLink",
                    "ImportMesh:="        , False
                ],
                "IsSbrRangeDoppler:="    , True,
                "SbrRangeDopplerWaveformType:=", "PulseDoppler",
                "SbrRangeDopplerTimeVariable:=", self.time_var_name,
                "SbrRangeDopplerCenterFreq:=", f"{simulation_params['sol_freq']}GHz",
                "SbrRangeDopplerRangeResolution:=", f"{simulation_params['range_res']}meter",
                "SbrRangeDopplerRangePeriod:=", f"{simulation_params['range_period']}meter",
                "SbrRangeDopplerVelocityResolution:=", f"{simulation_params['vel_res']}m_per_sec",
                "SbrRangeDopplerVelocityMin:=", f"{simulation_params['vel_min']}m_per_sec",
                "SbrRangeDopplerVelocityMax:=", f"{simulation_params['vel_max']}m_per_sec",
                "DopplerRayDensityPerWavelength:=", simulation_params['ray_density'],
                "MaxNumberOfBounces:="    , simulation_params['bounces']
            ])

        return setup_name
    
    def insert_parametric_sweep(self,time_start,time_stop,time_step,setup_name):
            """
            create parametric sweep setup for the time values specified in the file
            exported from scanner for each time step

            returns name of parametric sweep
            """
            oModule = self.oDesign.GetModule("Optimetrics")
            sweep_str = "LIN " + str(time_start) + "s " + str(time_stop) + "s " + str(time_step) + "s"
            para_sweep_name = "Full_Time_Sweep"
            original_name = para_sweep_name
            all_para_setup_names = oModule.GetSetupNames()
            n=1
            while para_sweep_name in all_para_setup_names:
                para_sweep_name = original_name + str(n)
                n+=1
            oModule = self.oDesign.GetModule("Optimetrics")
            oModule.InsertSetup("OptiParametric", 
                [
                    "NAME:"+para_sweep_name,
                    "IsEnabled:="        , True,
                    [
                        "NAME:ProdOptiSetupDataV2",
                        "SaveFields:="        , False
                    ],
                    [
                        "NAME:StartingPoint"
                    ],
                    "Sim. Setups:="        , [setup_name],
                    [
                        "NAME:Sweeps",
                        [
                            "NAME:SweepDefinition",
                            "Variable:="        , self.time_var_name,
                            "Data:="        , sweep_str,
                            "OffsetF1:="        , False,
                            "Synchronize:="        , 0
                        ]
                    ],
                    [
                        "NAME:Sweep Operations"
                    ],
                    [
                        "NAME:Goals"
                    ]
                ])

            return para_sweep_name