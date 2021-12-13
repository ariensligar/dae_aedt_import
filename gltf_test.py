# -*- coding: utf-8 -*-
"""
Created on Thu Dec  9 11:29:17 2021

@author: asligar
"""
from pygltflib import GLTF2
import pyvista as pv
import pyrender
filename = "C:/Users/asligar/OneDrive - ANSYS, Inc/Desktop/delete/gltf/gltf/scene.gltf"

gltf = GLTF2().load(filename)

current_scene = gltf.scenes[gltf.scene]
node_index = current_scene.nodes[0]  # scene.nodes is the indices, not the objects 
box = gltf.nodes[node_index]
box.matrix  # will output vertices for the box object


#pyrender.Viewer(gltf.scene)
#pyrender.Viewer(filename, use_raymond_lighting=True)
#pl.set_environment_texture(texture)

mesh_gtlf = gltf.meshes[gltf.scenes[gltf.scene].nodes[0]]

import vtk

def load_gltf(filename,
                renderer=None,
                opacity=1.0,
                specular=0.1,
                ambient=0.0,
                scale=(1, 1, 1),
                translate=(0, 0, 0),
                rotate=(0, 0, 0),
                mesh_color="blue",
                use_wireframe=False,
                scale_then_translate=False):

    colors = vtk.vtkNamedColors()


    reader = vtk.vtkGLTFReader()
    reader.SetFileName(filename)

    reader.Update()

    polydata = vtk.vtkCompositeDataGeometryFilter()
    polydata.SetInputConnection(reader.GetOutputPort())

    transform = vtk.vtkTransform()
    transform.RotateX(rotate[0])
    transform.RotateY(rotate[1])
    transform.RotateZ(rotate[2])

    if scale_then_translate:
        transform.Scale(scale)
        transform.Translate(translate)
    else:
        transform.Translate(translate)
        transform.Scale(scale)

    transformFilter = vtk.vtkTransformPolyDataFilter()
    transformFilter.SetInputConnection(polydata.GetOutputPort())
    transformFilter.SetTransform(transform)
    transformFilter.Update()


    mapper = vtk.vtkPolyDataMapper()

    mapper.SetInputConnection(0, transformFilter.GetOutputPort(0))

    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    if mesh_color is not None:
        actor.GetProperty().SetColor(colors.GetColor3d(mesh_color))
    actor.GetProperty().SetOpacity(opacity)


    actor.GetProperty().SetSpecular(specular)
    actor.GetProperty().SetSpecularPower(80.0)
    actor.GetProperty().SetAmbient(ambient)

    actor.GetProperty().SetInterpolationToGouraud()

    if use_wireframe:
        actor.GetProperty().SetRepresentationToWireframe()

    if renderer is None:
        return actor
    else:
        renderer.AddActor(actor)
        
test = load_gltf(filename)