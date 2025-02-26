from .renderer import Renderer, get_global_cameras
from .pysmpl import PySMPL
smpl = PySMPL()
import torch
renderer = None
global_R = None
global_lights = None
cameras = None

def render_image(verts, pc=None, ground_height=0.5,
                 transform = torch.tensor([[-1,0,0],[0,0,1],[0,1,0]]).float(),
                 ground_position = (0, 0)):
    global renderer
    global global_R
    global global_lights
    global cameras
    if renderer is None:
        width = 1920
        height = 1080
        focal_length = (width ** 2 + height ** 2) ** 0.5
        renderer = Renderer(width, height, focal_length,verts.device, smpl.faces)
        renderer.set_ground(50, center_x = ground_position[0], center_z = ground_position[1])
    _verts = verts.reshape(-1, 3) @ transform
    _pc = None
    if pc is not None:
        _pc = pc.reshape(-1, 3) @ transform
        _pc[..., 1] += ground_height
    _verts[..., 1] += ground_height
    if global_R is None:
        global_R, global_T, global_lights = get_global_cameras(verts.view(1, -1, 3), verts.device, 15)
        cameras = renderer.create_camera(global_R[0], global_T[0])
    faces = renderer.faces.clone().squeeze(0)
    colors = torch.ones((1, 4)).float().to(verts.device)
    colors[..., :3] *= 0.9
    img_glob = renderer.render_with_ground(verts,
                                        faces, colors, cameras, global_lights, _pc)
    return img_glob

def clear_renderer():
    global renderer
    global global_R
    renderer = None
    global_R = None
    





    




