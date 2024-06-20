# 可视化眼动数据：白模
# 输入：
#   处理过的eyepose和pose的txt数据目录
#   可视化结果输出目录
# 输出：
#   可视化视频（参数选择是否需要pose）
import os, sys
import cv2
import numpy as np
from time import time
from scipy.io import savemat
import argparse
from tqdm import tqdm
import torch
import pickle
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from decalib.deca import DECA
from decalib.datasets import datasets
from decalib.utils import util
from decalib.utils.config import cfg as deca_cfg
from decalib.utils.tensor_cropper import transform_points
import subprocess
import os
import json
import re
import math
os.environ['PYOPENGL_PLATFORM'] = 'egl'  # egl
import pyrender
import trimesh
from psbody.mesh import Mesh
import tempfile
from subprocess import call
from datetime import datetime

processed_txt_base='/media/lenovo/本地磁盘/1_talkingface/eye_move_data/data_process/'
eye_pose_txt_base = '/media/lenovo/本地磁盘/1_talkingface/eye_move_data/data_process/eye_txt_output'
pose_txt_base = '/media/lenovo/本地磁盘/1_talkingface/eye_move_data/data_process/pose_txt_output'
vis_result_base="/media/lenovo/本地磁盘/1_talkingface/eye_move_data/eye_move_results"

# todo:加上参数（是否要pose）
if True:
    vis_result_name = "ye_data_vis_wo_pose"
    vis_result_path="/media/lenovo/本地磁盘/1_talkingface/eye_move_data/eye_move_results/eye_data_vis_w_pose"
# else:
#     vis_result_name = "ye_data_vis_wo_pose"
#     vis_result_path = os.path.join(vis_result_base, vis_result_name)

# 读取眼动数据，更新该帧的codedict['eye_pose']
def import_eye(group_path, iframe):
    # group_path：眼动数据(0-1-a_raw)
    # iframe:(帧数)
    group_name = os.path.basename(group_path)
    formatted_iframe = f"{int(iframe):04d}"

    # 构建文件名
    eye_frame = os.path.join(group_path, f'{group_name}_{formatted_iframe}.txt')
    # 读取每帧的 txt 文件中的数据
    with open(eye_frame, 'r') as file:
        json_data = file.read()
    eye_data = json.loads(json_data)

    gaze_r = eye_data.get('gaze_r')
    gaze_l = eye_data.get('gaze_l')
    yaw_r = gaze_r[0]
    pitch_r = gaze_r[1]
    yaw_l = gaze_l[0]
    pitch_l = gaze_l[1]

    return torch.tensor(
        [[pitch_r, yaw_r, 0, pitch_l, yaw_l, 0]], dtype=torch.float32)

# 读取pose数据(imu.csv)，更新该帧的codedict['pose']
def import_pose(ori_pose, group_path, iframe):
    group_name = os.path.basename(group_path)
    formatted_iframe = f"{int(iframe):04d}"

    # 构建文件名
    pose_frame = os.path.join(group_path, f'{group_name}_{formatted_iframe}.txt')
    # 读取 txt 文件中的数据
    with open(pose_frame, 'r') as file:
        json_data = file.read()

    pose_data = json.loads(json_data)

    pitch = pose_data.get('pitch_deg')
    roll = pose_data.get('roll_deg')
    yaw = pose_data.get('yaw_deg')

        # 将角度转换为弧度
    pitch_rad = math.radians(pitch)
    yaw_rad = math.radians(yaw)
    roll_rad = math.radians(roll)
        # 创建一个形状为 1x3 的张量，包含 pitch、yaw 和 roll 的弧度值
    pose_rad_tensor = torch.tensor([[-1*pitch_rad, yaw_rad, roll_rad]]).cuda()

    return torch.cat((pose_rad_tensor, ori_pose[:, 3:]), dim=1)

#渲染函数
def render_mesh_helper(mesh, t_center, rot=np.zeros(3), tex_img=None, z_offset=0):
    # 法线贴图
    # 1. 读取 PNG 文件
    # normal_map_path = '/media/lenovo/本地磁盘/1_talkingface/DECA/TestSamples/examples/results/IMG_0392_inputs/IMG_0392_inputs_normals.png'
    # normal_map_image = Image.open(normal_map_path)
    # normal_texture = pyrender.Texture(source=normal_map_image, source_channels='RGB')

    camera_params = {'c': np.array([400, 400]),
                     'k': np.array([-0.19816071, 0.92822711, 0, 0, 0]),
                     'f': np.array([4754.97941935 / 2, 4754.97941935 / 2])}
    # 中心坐标（c）、畸变系数（k）和焦距（f）

    frustum = {'near': 0.01, 'far': 3.0, 'height': 800, 'width': 800}
    # 定义视锥体，包括近平面和远平面、高度和宽度。（渲染到2D平面）

    ########################################################################################################################
    # 函数创建输入网格的副本，并应用旋转变换。这个变换包括使用 cv2.Rodrigues 将旋转向量 rot 转换成旋转矩阵，并将其应用于网格的顶点。
    mesh_copy = Mesh(mesh.v, mesh.f)
    mesh_copy.v[:] = cv2.Rodrigues(rot)[0].dot((mesh_copy.v - t_center).T).T + t_center

    intensity = 2.0
    # 使用 pyrender 的 MetallicRoughnessMaterial 定义了一个原始材质。
    primitive_material = pyrender.material.MetallicRoughnessMaterial(  # 这个类才是PBR渲染
        alphaMode='BLEND',
        baseColorFactor=[0.3, 0.3, 0.3, 1.0],
        metallicFactor=0.8,
        roughnessFactor=0.8,
        # baseColorTexture=rgba_data,
        # normalTexture=normal_texture,
        wireframe=True,
    )

    # 使用 pyrender.Mesh.from_trimesh 从变换后的网格中创建可渲染的网格。
    v_eye = mesh_copy.v[:]
    tri_mesh = trimesh.Trimesh(vertices=v_eye, faces=mesh_copy.f)
    # tri_mesh.visual.vertex_colors = np.ones((5023, 4)) * [255,255,255,1]

    # TODO from github
    # Placeholder texture
    texture_image = np.ones((1, 1, 3), dtype=np.uint8) * 255

    # Create a Trimesh texture
    texture = trimesh.visual.texture.TextureVisuals(
        uv=(tri_mesh.vertices[:, :2] - np.min(tri_mesh.vertices[:, :2], axis=0)) / np.ptp(tri_mesh.vertices[:, :2],
                                                                                          axis=0),
        image=texture_image
    )

    # Set the texture for the mesh
    tri_mesh.visual = texture

    # 修改渲染网格的创建，将纹理应用到网格上

    render_mesh = pyrender.Mesh.from_trimesh(tri_mesh, material=primitive_material, smooth=True)

    # TODO eyes sphere
    pts = [mesh_copy.v[4051], mesh_copy.v[4597]]

    sm = trimesh.creation.uv_sphere(radius=0.002)
    sm.visual.vertex_colors = [0.5, 0.5, 0.5]
    tfs = np.tile(np.eye(4), (len(pts), 1, 1))
    tfs[:, :3, 3] = pts
    eyes = pyrender.Mesh.from_trimesh(sm, poses=tfs)

    # 场景配置：根据参数 args.background_black，设置场景的环境光和背景颜色。
    scene = pyrender.Scene(ambient_light=[.2, .2, .2], bg_color=[0, 0, 0])
    # if args.background_black:
    #     scene = pyrender.Scene(ambient_light=[.2, .2, .2], bg_color=[0, 0, 0])
    # else:
    #     scene = pyrender.Scene(ambient_light=[.2, .2, .2], bg_color=[255, 255, 255])
    # 还创建了一个相机，并使用之前定义的内部参数将其添加到场景中。
    camera = pyrender.IntrinsicsCamera(fx=camera_params['f'][0],
                                       fy=camera_params['f'][1],
                                       cx=camera_params['c'][0],
                                       cy=camera_params['c'][1],
                                       znear=frustum['near'],
                                       zfar=frustum['far'])

    scene.add(render_mesh, pose=np.eye(4))
    scene.add(eyes, pose=np.eye(4))

    # 相机姿态和照明
    camera_pose = np.eye(4)
    camera_pose[:3, 3] = np.array([0, 0, 1.0 - z_offset])
    scene.add(camera, pose=[[1, 0, 0, 0],
                            [0, 1, 0, 0],
                            [0, 0, 1, 1],
                            [0, 0, 0, 1]])

    angle = np.pi / 6.0
    pos = camera_pose[:3, 3]
    light_color = np.array([1., 1., 1.])
    light = pyrender.DirectionalLight(color=light_color, intensity=intensity)

    light_pose = np.eye(4)
    light_pose[:3, 3] = pos
    scene.add(light, pose=light_pose.copy())

    light_pose[:3, 3] = cv2.Rodrigues(np.array([angle, 0, 0]))[0].dot(pos)
    scene.add(light, pose=light_pose.copy())

    light_pose[:3, 3] = cv2.Rodrigues(np.array([-angle, 0, 0]))[0].dot(pos)
    scene.add(light, pose=light_pose.copy())

    light_pose[:3, 3] = cv2.Rodrigues(np.array([0, -angle, 0]))[0].dot(pos)
    scene.add(light, pose=light_pose.copy())

    light_pose[:3, 3] = cv2.Rodrigues(np.array([0, angle, 0]))[0].dot(pos)
    scene.add(light, pose=light_pose.copy())

    flags = pyrender.RenderFlags.SKIP_CULL_FACES | pyrender.RenderFlags.FACE_NORMALS

    # 函数使用 pyrender.OffscreenRenderer 渲染场景
    try:
        r = pyrender.OffscreenRenderer(viewport_width=frustum['width'], viewport_height=frustum['height'])
        color, _ = r.render(scene, flags=flags)
        placeholder = 1
    except Exception as e:
        print(e)
        print('pyrender: Failed rendering frame')
        color = np.zeros((frustum['height'], frustum['width'], 3), dtype='uint8')

    # 返回渲染的彩色图像，颜色通道被反转（color[..., ::-1]），以将其从 BGR（OpenCV中常见）转换为 RGB 格式
    return color[..., ::-1]


def main(args):

# 获取当前日期和时间
    formatted_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M")
    output_path = os.path.join(vis_result_path, f"{vis_result_path}_{formatted_datetime}")
    os.makedirs(output_path, exist_ok=True)
# 一个group是一次采集
    for group in os.listdir(eye_pose_txt_base):
        eye_group_path = os.path.join(eye_pose_txt_base, group)  # 获取文件夹完整路径
        pose_group_path = os.path.join(pose_txt_base, group)  # 获取文件夹完整路径
        zeroed_pose = torch.zeros((1, 6)).cuda()

        #从输入选取主体id
        # if args.rasterizer_type != 'standard':
        #     args.render_orig = False
        subject_clip = args.inputpath.split('/')[-1].split('.')[0]
        device = args.device
        # load test images
        testdata = datasets.TestData(args.inputpath, iscrop=args.iscrop, face_detector=args.detector,
                                     sample_step=args.sample_step)
        # run DECA
        deca_cfg.model.use_tex = args.useTex
        deca_cfg.rasterizer_type = args.rasterizer_type
        deca_cfg.model.extract_tex = args.extractTex
        deca = DECA(config=deca_cfg, device=device)

        # render 眼睛相关
        template = Mesh(filename='/media/lenovo/本地磁盘/1_talkingface/template_vox/FLAME_sample.ply')

        tmp_video_file = tempfile.NamedTemporaryFile('w', suffix='.mp4', dir=output_path)
        writer = cv2.VideoWriter(tmp_video_file.name, cv2.VideoWriter_fourcc(*'mp4v'), 29, (800, 800), True)

        for i in tqdm(range(len(testdata))):  # tqdm是一个进度条,input图像个数
            name = testdata[i]['imagename']
            images = testdata[i]['image'].to(device)[None, ...]
            with torch.no_grad():
                codedict = deca.encode(images)
                # 计算该次采集记录的样本个数（以眼动数据为准）
                files_and_subdirectories = os.listdir(eye_group_path)
                num_files_and_subdirectories = len(files_and_subdirectories)

                for n_th_data in tqdm(range(num_files_and_subdirectories)):
                    # 根据采集数据更新每一帧的eyepose和pose
                    codedict['eye_pose'] = import_eye(eye_group_path, n_th_data)
                    codedict['pose'] = import_pose(zeroed_pose, pose_group_path, n_th_data)
                    #decode
                    opdict, visdict = deca.decode(codedict, iframe=i, name=subject_clip)  # tensor
                    # render
                    center = np.mean(opdict['verts'][0].cpu().numpy(), axis=0)
                    render_mesh = Mesh(opdict['verts'][0].cpu().numpy(), template.f)
                    pred_img = render_mesh_helper(render_mesh, center)
                    pred_img = pred_img.astype(np.uint8)
                    writer.write(pred_img)

                writer.release()
                file_name = group

                video_fname = os.path.join(output_path, file_name + '.mp4')
                cmd = ('ffmpeg' + ' -i {0} -pix_fmt yuv420p -q:v 0 {1}'.format(
                    tmp_video_file.name, video_fname)).split()
                call(cmd)
                print("******cmd1 finished")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DECA: Detailed Expression Capture and Animation')

    parser.add_argument('-i', '--inputpath', default='TestSamples/examples/IMG_0392_inputs.jpg', type=str,
                        help='path to the test data, can be image folder, image path, image list, video')
    parser.add_argument('-s', '--savefolder', default='TestSamples/examples/results', type=str,
                        help='path to the output directory, where results(obj, txt files) will be stored.')
    parser.add_argument('--device', default='cuda', type=str,
                        help='set device, cpu for using cpu')
    # process test images
    parser.add_argument('--iscrop', default=True, type=lambda x: x.lower() in ['true', '1'],
                        help='whether to crop input image, set false only when the test image are well cropped')
    parser.add_argument('--sample_step', default=10, type=int,
                        help='sample images from video data for every step')
    parser.add_argument('--detector', default='fan', type=str,
                        help='detector for cropping face, check decalib/detectors.py for details')
    # rendering option
    parser.add_argument('--rasterizer_type', default='standard', type=str,
                        help='rasterizer type: pytorch3d or standard')
    parser.add_argument('--render_orig', default=False, type=lambda x: x.lower() in ['true', '1'],
                        help='whether to render results in original image size, currently only works when rasterizer_type=standard')
    # save
    parser.add_argument('--useTex', default=False, type=lambda x: x.lower() in ['true', '1'],
                        help='whether to use FLAME texture model to generate uv texture map, \
                            set it to True only if you downloaded texture model')
    parser.add_argument('--extractTex', default=True, type=lambda x: x.lower() in ['true', '1'],
                        help='whether to extract texture from input image as the uv texture map, set false if you want albeo map from FLAME mode')
    parser.add_argument('--saveVis', default=False, type=lambda x: x.lower() in ['true', '1'],
                        help='whether to save visualization of output')
    parser.add_argument('--saveKpt', default=False, type=lambda x: x.lower() in ['true', '1'],
                        help='whether to save 2D and 3D keypoints')
    parser.add_argument('--saveDepth', default=False, type=lambda x: x.lower() in ['true', '1'],
                        help='whether to save depth image')
    parser.add_argument('--saveObj', default=False, type=lambda x: x.lower() in ['true', '1'],
                        help='whether to save outputs as .obj, detail mesh will end with _detail.obj. \
                            Note that saving objs could be slow')
    parser.add_argument('--saveMat', default=False, type=lambda x: x.lower() in ['true', '1'],
                        help='whether to save outputs as .mat')
    parser.add_argument('--saveImages', default=False, type=lambda x: x.lower() in ['true', '1'],
                        help='whether to save visualization output as seperate images')
    parser.add_argument('--useVideo', default=False, type=lambda x: x.lower() in ['true', '1'],
                        help='whether to save visualization output as seperate images')
    parser.add_argument('--saveSeq', default=False, type=lambda x: x.lower() in ['true', '1'],
                        help='whether to save video vertices sequence')
    parser.add_argument('--visEye', default=True, type=lambda x: x.lower() in ['true', '1'],
                        help='whether to add albedo to mesh')
    parser.add_argument('-p', '--pickle', default='/media/lenovo/本地磁盘/1_talkingface/template_vox/pickle/',
                        type=str,
                        help='where to save template')
    parser.add_argument('--demo', default=True, type=lambda x: x.lower() in ['true', '1'],
                        help='whether to output demo image')
    parser.add_argument('--final', default=True, type=lambda x: x.lower() in ['true', '1'],
                        help='whether to output demo image')
    main(parser.parse_args())
