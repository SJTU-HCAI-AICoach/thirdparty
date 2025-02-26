import numpy as np
import copy
import random
import torch
import os
import plotly.graph_objects as go # or plotly.express as px
from dash import Dash, dcc, html
import argparse
import torch.nn.functional as F
from .pysmpl import PySMPL
from .smpl.SMPL import ModelOutput
smpl = PySMPL()
   
class Visualizer:
    colors = ["rgb(255, 0, 0)",
              "rgb(0, 255, 0)",
              "rgb(0, 0, 255)",
              "rgb(85, 26, 13)",      # 深棕色  
    "rgb(34, 139, 34)",     # 深森林绿  
    "rgb(0, 64, 128)",      # 深海军蓝（与上一个略有不同，更偏向蓝色）  
    "rgb(139, 0, 139)",     # 深李子色  
    "rgb(105, 105, 105)",   # 深灰色  
    "rgb(139, 69, 19)",     # 深马鞍棕色  
    "rgb(69, 19, 139)",]
    line_colors = [  
        "rgb(100, 0, 0)",       # 深红色  
        "rgb(0, 100, 0)",       # 深绿色  
        "rgb(0, 0, 100)",       # 深蓝色  
        "rgb(128, 0, 128)",     # 深紫色  
        "rgb(0, 128, 128)",     # 深青色  
        "rgb(128, 128, 0)",     # 深橄榄色  
        "rgb(128, 0, 0)",       # 暗红色  
        "rgb(0, 128, 0)",       # 暗绿色  
        "rgb(75, 0, 130)",      # 深靛蓝色  
        "rgb(64, 0, 64)"        # 深海军蓝  
    ]
    def __init__(self):
        pass
    @staticmethod
    def show_points(points, t = 0, fix_axis = False):
        connections = torch.tensor([[ 1,  4],[ 1,  0],
            [ 2,  5],[ 2,  0],[ 3,  6],[ 3,  0],[ 4,  7],
            [ 5,  8],[ 6,  9],[ 7, 10],[ 8, 11],[ 9, 12],
            [12, 13],[12, 14],[12, 15],[13, 16],[14, 17],
            [16, 18],[17, 19],[18, 20],[19, 21],[20, 22],
            [21, 23]], dtype=torch.long)
        faces = np.array(smpl.faces)
        meshes = []
        for idx, _points in enumerate(points):
            is_mesh = False
            is_skeleton = False
            if hasattr(_points, "vertices"):
                _points = _points.vertices
                is_mesh = True
            num_points = _points.shape[-2]
            if num_points == 6890:
                is_mesh = True
            if num_points == 24:
                is_skeleton = True
            _points = _points.view(-1, num_points, 3).cpu().numpy()
            if t < 0:
                t = random.randint(0, len(_points) - 1)
            pc = _points[t]
            lines = []
            if not is_mesh:
                pc_show = go.Scatter3d(x=pc[:,0],y=pc[:,1],z=pc[:,2],mode="markers",marker=dict(size=4, color=Visualizer.colors[idx], opacity=0.8))
                if is_skeleton:
                    for start, end in connections:  
                        line = go.Scatter3d(  
                            x=[pc[start, 0], pc[end, 0]],  
                            y=[pc[start, 1], pc[end, 1]],  
                            z=[pc[start, 2], pc[end, 2]],  
                            mode='lines',  
                            line=dict(  
                                color=Visualizer.line_colors[idx],
                                width=2  
                            )  
                        )
                        lines.append(line)
                    
            else:
                pc_show = go.Mesh3d(
                    x=pc[:, 0],
                    y=pc[:, 1],
                    z=pc[:, 2],
                    i=faces[:, 0],
                    j=faces[:, 1],
                    k=faces[:, 2],
                    color=Visualizer.colors[idx],
                    opacity=0.5,
                )
            meshes.append(pc_show)
            if not(is_mesh) and is_skeleton:
                meshes.extend(lines)
                
        if fix_axis:
            layout = go.Layout(  
                title='Mesh3d Plot with Axis Range Settings',  
                scene=dict(  
                    xaxis=dict(range=[-1.1, 1.1]), 
                    yaxis=dict(range=[-1.1, 1.1]), 
                    zaxis=dict(range=[-1.1, 1.1])   ,
                    aspectmode='cube',
                    aspectratio=dict(x=1, y=1, z=1)
                )  
            ) 
        else:
            layout = go.Layout(  
                title='Mesh3d Plot with Axis Range Settings',  
                scene=dict(  
                    aspectmode='cube',
                    aspectratio=dict(x=1, y=1, z=1)
                )  
            )
        fig = go.Figure(data = meshes, layout=layout)
        app = Dash()
        app.layout = html.Div([
            dcc.Graph(figure=fig)
        ])

        app.run_server(debug=True, use_reloader=False)
                    
    @staticmethod 
    def show_skeletons(points, t = 0, fix_axis = False) :
        connections = torch.tensor([[ 1,  4],[ 1,  0],
            [ 2,  5],[ 2,  0],[ 3,  6],[ 3,  0],[ 4,  7],
            [ 5,  8],[ 6,  9],[ 7, 10],[ 8, 11],[ 9, 12],
            [12, 13],[12, 14],[12, 15],[13, 16],[14, 17],
            [16, 18],[17, 19],[18, 20],[19, 21],[20, 22],
            [21, 23]], dtype=torch.long)
        meshes = []
        for idx, _points in enumerate(points):
            if type(_points) is ModelOutput:
                _points = _points.joints
            num_points = _points.shape[-2]
            _points = _points.view(-1, num_points, 3).cpu().numpy()
            pc = _points[t]
            pc_show = go.Scatter3d(x=pc[:,0],y=pc[:,1],z=pc[:,2],mode="markers",marker=dict(size=3, color=Visualizer.colors[idx], opacity=0.8))
            meshes.append(pc_show)
            
            for start, end in connections:  
                line = go.Scatter3d(  
                    x=[pc[start, 0], pc[end, 0]],  
                    y=[pc[start, 1], pc[end, 1]],  
                    z=[pc[start, 2], pc[end, 2]],  
                    mode='lines',  
                    line=dict(  
                        color=Visualizer.line_colors[idx],
                        width=1  
                    )  
                )
                meshes.append(line)
        if fix_axis:
            layout = go.Layout(  
                title='Mesh3d Plot with Axis Range Settings',  
                scene=dict(  
                    xaxis=dict(range=[-1.1, 1.1]), 
                    yaxis=dict(range=[-1.1, 1.1]), 
                    zaxis=dict(range=[-1.1, 1.1])   ,
                    aspectmode='cube',
                    aspectratio=dict(x=1, y=1, z=1)
                )  
            ) 
        else:
            layout = go.Layout(  
                title='Mesh3d Plot with Axis Range Settings',  
                scene=dict(  
                    aspectmode='cube',
                    aspectratio=dict(x=1, y=1, z=1)
                )  
            )
        fig = go.Figure(data = meshes, layout=layout)
        app = Dash()
        app.layout = html.Div([
            dcc.Graph(figure=fig)
        ])

        app.run_server(debug=True, use_reloader=False)
    @staticmethod 
    def show_seqs(smpls):
        colors = ["rgb(255, 0, 0)", "rgb(0, 255, 0)", "rgb(0, 0, 255)"]
        connections = torch.tensor([[ 1,  4],[ 1,  0],
            [ 2,  5],[ 2,  0],[ 3,  6],[ 3,  0],[ 4,  7],
            [ 5,  8],[ 6,  9],[ 7, 10],[ 8, 11],[ 9, 12],
            [12, 13],[12, 14],[12, 15],[13, 16],[14, 17],
            [16, 18],[17, 19],[18, 20],[19, 21],[20, 22],
            [21, 23]], dtype=torch.long)
        assert type(smpls) is list
        first_smpl = smpls[0]

        if hasattr(first_smpl, "vertices"):
            _vs = first_smpl.vertices.cpu().numpy()
        else:
            _vs = first_smpl.cpu().numpy() 
        faces = np.array(smpl.faces)
        frames = []
        data_first = None
        for T in range(_vs.shape[0]):
            meshes = []
            for idx, _smpl in enumerate(smpls):
                if hasattr(_smpl, "vertices"):
                    _vs = _smpl.vertices.cpu().numpy()
                else:
                    _vs = _smpl.cpu().numpy() 

                gt_verts = _vs[T]
                if len(gt_verts) == 6890:
                    mesh_gt_show = go.Mesh3d(
                        x=gt_verts[:, 0],
                        y=gt_verts[:, 1],
                        z=gt_verts[:, 2],
                        i=faces[:, 0],
                        j=faces[:, 1],
                        k=faces[:, 2],
                        color=colors[idx],
                        opacity=0.5,
                    )
                    meshes.append(mesh_gt_show)
                elif len(gt_verts) == 24:
                    for start, end in connections:  
                        line = go.Scatter3d(  
                            x=[gt_verts[start, 0], gt_verts[end, 0]],  
                            y=[gt_verts[start, 1], gt_verts[end, 1]],  
                            z=[gt_verts[start, 2], gt_verts[end, 2]],  
                            mode='lines',  
                            line=dict(  
                                color=Visualizer.line_colors[idx],
                                width=4 
                            )  
                        )
                        meshes.append(line)
                    pc_show = go.Scatter3d(x=gt_verts[:,0],y=gt_verts[:,1],z=gt_verts[:,2],mode="markers",marker=dict(size=3, color=Visualizer.colors[idx], opacity=0.8))
                    meshes.append(pc_show)
                else:
                    pc_show = go.Scatter3d(x=gt_verts[:,0],y=gt_verts[:,1],z=gt_verts[:,2],mode="markers",marker=dict(size=3, color=Visualizer.colors[idx], opacity=0.8))
                    meshes.append(pc_show)
            frame = go.Frame(
                data = meshes,
                name="frame" + str(T)
            )
            frames.append(frame)
            if T == 0:
                data_first = copy.deepcopy(meshes)
        layout = go.Layout(  
            title='Mesh3d Plot with Axis Range Settings',  
            # scene=dict(  
            #     xaxis=dict(range=[-1.1, 1.1]), 
            #     yaxis=dict(range=[-1.1, 1.1]), 
            #     zaxis=dict(range=[-1.1, 1.1])   ,
            #     aspectmode='cube',
            #     aspectratio=dict(x=1, y=1, z=1)
            # )  
        ) 
        fig = go.Figure(data=data_first, layout=layout)
        fig.frames = frames

        fig.update_layout(
            sliders=[dict(
                active=0,
                currentvalue={"prefix": "Frame: "},
                pad={"t": 50},
                steps=[
                    dict(method="animate", args=[[frame.name], dict(frame=frame, transition=dict(duration=0))])
                    for frame in frames
                ]
            )]
        )


        app = Dash()
        app.layout = html.Div([
            dcc.Graph(figure=fig)
        ])

        app.run_server(debug=True, use_reloader=False)
        app.run_server(debug=True, use_reloader=False)

