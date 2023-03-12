#  Copyright (c) 2022. Kwon. Sweetk LAB., All rights reserved.
#  Kwon Hyeokmin <hykwon8952@gmail.com>
#  Sweetk Ltd. <sweetk_lab@sweetk.co.kr>

import os
# local
try:
    os.environ['PYOPENGL_PLATFORM'] = 'osmesa'
    import pyrender
except:
    os.environ.pop('PYOPENGL_PLATFORM')
    import pyrender
import numpy as np
import colorsys
import trimesh


class Renderer(object):

    def __init__(self, viewport_width, viewport_height):
        self.scene = pyrender.Scene(ambient_light=(.3, .3, .3))
        # renderer
        self.renderer = pyrender.OffscreenRenderer(
            viewport_width=viewport_width, viewport_height=viewport_height, point_size=1.0)

    def insert_camera(self, cam_param):
        focal, princpt, cam_no = cam_param['focal'], cam_param['princpt'], cam_param['cam_no']
        camera = pyrender.IntrinsicsCamera(fx=focal, fy=focal, cx=princpt[0], cy=princpt[1])
        cam_m = np.linalg.inv(cam_param['extrinsics'])
        cam_m[:, 1:3] *= -1
        self.scene.add(camera, 'cam_{}'.format(cam_no), cam_m)
        # light
        light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=1.)
        self.scene.add(light, 'light_{}'.format(cam_no), cam_m)
        self.scene.add(light, 'light_{}'.format(cam_no))

    def update_mesh(self, mesh):
        mesh_color = colorsys.hsv_to_rgb(0.6, 0.5, 1.0)
        material = pyrender.MetallicRoughnessMaterial(metallicFactor=.0, alphaMode='OPAQUE', baseColorFactor=mesh_color)
        node = self.scene.get_nodes(name='mesh')
        if len(node) > 1:
            self.scene.remove_node(node.pop())
        mesh = pyrender.Mesh.from_trimesh(mesh, material=material, smooth=False)
        self.scene.add(mesh, 'mesh')

    def render_mesh(self, cam_no, bg):
        # render
        cam_name = 'cam_{}'.format(cam_no)
        cam_node = self.scene.get_nodes(name=cam_name)

        self.scene.main_camera_node = cam_node.pop()
        rgb, depth = self.renderer.render(self.scene, flags=pyrender.RenderFlags.RGBA)
        rgb = rgb[:,:,:3].astype(np.float32)
        valid_mask = (depth > 0)[:,:,None]

        # save to image
        img = rgb * valid_mask + bg * (1-valid_mask)
        return img, valid_mask


def display(visimg, vertices, focal, princpt, extrinsics, viewport_width, viewport_height, faces):
    mesh = trimesh.Trimesh(vertices, faces)

    img = np.copy(visimg)
    renderer = Renderer(viewport_width, viewport_height)
    camera_no = 0

    cam_param = {'focal': focal,
                 'princpt': princpt,
                 'extrinsics': extrinsics,
                 'cam_no': camera_no}
    renderer.insert_camera(cam_param=cam_param)
    renderer.update_mesh(mesh)

    img, _ = renderer.render_mesh(cam_no=camera_no, bg=img)
    del renderer

    return img
