import bpy
import numpy as np
import random
import os
import pandas as pd
from math import radians
from mathutils import Vector, Euler, Matrix
import time
from bpy_extras.object_utils import world_to_camera_view
import re

# ===================================================================
# 🚀 RTX 3070 GPU 强力配置
# ===================================================================
def setup_rtx3070_gpu():
    prefs = bpy.context.preferences.addons['cycles'].preferences
    prefs.get_devices()

    print("\n" + "="*50)
    print("检测到的计算设备:")
    for i, device in enumerate(prefs.devices):
        print(f"  [{i}] {device.name} | 类型: {device.type} | 启用: {device.use}")
    print("="*50 + "\n")

    gpu_enabled = False
    for device in prefs.devices:
        if device.type in ['CUDA', 'OPTIX']:
            device.use = True
            gpu_enabled = True
            print(f"✅ GPU已激活: {device.name} ({device.type})")

    if gpu_enabled:
        prefs.compute_device_type = 'OPTIX'
        bpy.context.scene.cycles.device = 'GPU'
        print("🎉 RTX 3070 GPU模式启用成功！使用OPTIX后端")
        return True
    else:
        bpy.context.scene.cycles.device = 'CPU'
        print("⚠️ 未检测到GPU设备，回退到CPU模式")
        return False


bpy.context.preferences.system.memory_cache_limit = 6144
setup_rtx3070_gpu()

print("\n" + "="*60)
print("🚀 卫星渲染系统 - 真实轨道FOV远景模式（含真实姿态）")
print("="*60 + "\n")


# 单体部件（你想单独做 YOLO 标签的物体）
# 例如 BUS、AO、天线……写真实名字即可
OBJ_NAMES = ['panel1', 'panel2', 'panel3', 'panel4','panel5','panel6','satellite1'] # 若无需求可以留空


# STK CSV 路径
STK_PATHS = {
    "OBS_POS": r"E:\jr\StkData\1.MSX-J2000 Position Velocity.csv",
    "OBS_ATT": r"E:\jr\StkData\1.MSX-Euler Angles.csv",
    "TGT_POS": r"E:\jr\StkData\1.CloudSat-J2000 Position Velocity.csv",
    "TGT_ATT": r"E:\jr\StkData\1.CloudSat-Euler Angles.csv",
}

# 输出目录

output_dir = r"E:\jr\SpaceTarget\1\output_real_fov"

# 渲染参数
FOV = 70
NUM = 100
START = 0
STEP = 20

# ===================================================================
# 🌍 STK真实轨道 + 姿态渲染类
# ===================================================================
class RealOrbitFOVRender:
    def __init__(self, saved_dir, fov=70):

        # 基础场景对象
        self.scene  = bpy.data.scenes['Scene']
        self.camera = bpy.data.objects['RenderCam']   # 新建的渲染相机
        self.axis   = bpy.data.objects['main_axis']
        self.light  = bpy.data.objects['Light']

        # ====== 创建/更新卫星几何中心 Empty 作为跟踪目标 ======
        sat = bpy.data.objects.get("satellite1")
        if sat is None or sat.type != 'MESH':
            print("⚠ 找不到 satellite1，使用 main_axis 作为跟踪目标")
            self.track_target = self.axis
        else:
            # 计算卫星包围盒几何中心（局部坐标）
            bb_center_local = sum((Vector(corner) for corner in sat.bound_box), Vector()) / 8.0
            bb_center_world = sat.matrix_world @ bb_center_local

            # 如果之前已经有 sat_center 就复用
            empty = bpy.data.objects.get("sat_center")
            if empty is None:
                empty = bpy.data.objects.new("sat_center", None)
                self.scene.collection.objects.link(empty)

            # 放到几何中心的位置
            empty.location = bb_center_world

            # 让空物体跟着卫星刚性运动：设为卫星的子物体
            empty.parent = sat
            empty.matrix_parent_inverse = sat.matrix_world.inverted()

            self.track_target = empty
            print("✅ 已创建/更新 sat_center Empty 作为跟踪目标")

        # ====== 设置 RenderCam & Track To 约束 ======
        self.scene.camera = self.camera  # 渲染也用这台相机

        # 清理 RenderCam（确保干净）
        self.camera.parent = None
        for c in list(self.camera.constraints):
            self.camera.constraints.remove(c)
        if self.camera.animation_data:
            self.camera.animation_data_clear()

        # 加一个 Track To：始终盯住 track_target
        track = self.camera.constraints.new(type='TRACK_TO')
        track.target     = self.track_target
        track.track_axis = 'TRACK_NEGATIVE_Z'  # 相机 -Z 指向目标
        track.up_axis    = 'UP_Y'              # Y 轴向上

        print("✅ RenderCam 已添加 Track To 约束，目标：", self.track_target.name)

        # 你要输出 bbox 的对象
        self.obj_names = OBJ_NAMES

        # 输出目录不变
        self.images_filepath = os.path.join(saved_dir, 'Data_Real_FOV')
        self.labels_filepath = os.path.join(saved_dir, 'Labels_Real_FOV')
        os.makedirs(self.images_filepath, exist_ok=True)
        os.makedirs(self.labels_filepath, exist_ok=True)

        # 渲染设置不变
        self.scene.render.engine = 'CYCLES'#启用光线追踪渲染引擎
        self.scene.render.image_settings.file_format = 'PNG'
        self.scene.cycles.samples = 128
        self.scene.cycles.device = 'GPU'
        self.scene.cycles.tile_size = 256

        # 分辨率保持原来的
        self.scene.render.resolution_x = 1280
        self.scene.render.resolution_y = 1280
        self.scene.render.resolution_percentage = 100#使用完整分辨率

        # 加载轨道与姿态
        self.load_stk_data()

        # 坐标系转化
        self.scale_factor = 0.001 #将STK的公里单位转换为Blender的米单位
        self.axis_conversion = Matrix.Rotation(radians(90), 4, 'X') #创建一个4×4旋转矩阵，绕X轴旋转90度
        #用途：解决STK坐标系（Z轴向上）与Blender坐标系（Y轴向上）的轴向不一致问题

        # FOV 不动
        self.camera_fov = fov
        self.camera.data.angle = radians(fov)

        # 创建标注对象
        self.objects = self.create_objects()

        # 刚性跟随 main_axis（不依赖父子层级）
        self.follow_objects, self.follow_local_mats = self.build_follow_mats()

    # =============================================================
    # 读取 STK 位置数据
  # =============================================================
    def read_stk_position(self, csv_path):
        """读取STK位置数据（极简版）"""
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"文件不存在: {csv_path}")
        
        # pandas自动识别分隔符和列名
        df = pd.read_csv(csv_path)
        
        # 如果列名带单位，清理一次
        df.columns = [c.split()[0] for c in df.columns]
        
        # 智能时间解析（自动处理所有格式）
        df['Time'] = pd.to_datetime(df['Time'], format='mixed', dayfirst=False, errors='coerce')
        
        # 验证数据
        if df.empty:
            raise ValueError(f"CSV文件为空或格式错误: {csv_path}")
        
        print(f"✅ 位置数据加载成功: {len(df)} 行")
        return df[['Time', 'x', 'y', 'z', 'vx', 'vy', 'vz']]

    def read_stk_euler(self, csv_path):
        """读取STK欧拉角（极简版）"""
        df = pd.read_csv(csv_path)
        
        # 清理列名
        df.columns = [c.split()[0] for c in df.columns]
        
        # 智能时间解析
        df['Time'] = pd.to_datetime(df['Time'], format='mixed', dayfirst=False, errors='coerce')
        
        if df.empty:
            raise ValueError(f"CSV文件为空或格式错误: {csv_path}")
        
        print(f"✅ 姿态数据加载成功: {len(df)} 行")
        return df[['Time', 'A', 'B', 'C']]
    # =============================================================
    # 合并 位置 + 姿态
    # =============================================================
    def merge_pos_att(self, pos_df, att_df):
        pos_df = pos_df.sort_values("Time")
        att_df = att_df.sort_values("Time")

        merged = pd.merge_asof(
            pos_df, att_df, on="Time",
            direction="nearest",
            tolerance=pd.Timedelta("1s")
        )

        merged = merged.dropna().reset_index(drop=True)
        print("位置帧数:", len(pos_df), "姿态帧数:", len(att_df), "合并后帧数:", len(merged))

        return merged

    # =============================================================
    # 总数据加载函数
    # =============================================================
    def load_stk_data(self):

        # ★★★★★ 你给的四个文件路径 ★★★★★
        obs_pos_path = STK_PATHS["OBS_POS"]
        obs_att_path = STK_PATHS["OBS_ATT"]
        tgt_pos_path = STK_PATHS["TGT_POS"]
        tgt_att_path = STK_PATHS["TGT_ATT"]

        # 读位置
        obs_pos = self.read_stk_position(obs_pos_path)
        tgt_pos = self.read_stk_position(tgt_pos_path)

        # 读姿态
        obs_att = self.read_stk_euler(obs_att_path)
        tgt_att = self.read_stk_euler(tgt_att_path)

        # 时间对齐（merge_asof）
        self.obs_data = self.merge_pos_att(obs_pos, obs_att)
        self.tgt_data = self.merge_pos_att(tgt_pos, tgt_att)

        print(f"MSX数据帧数: {len(self.obs_data)}")
        print(f"CloudSat数据帧数: {len(self.tgt_data)}")

    # =============================================================
    # STK 转 Blender 坐标 + 姿态
    # =============================================================
    def convert_stk_to_blender(self, row):
        pos = Vector((row['x'], row['y'], row['z'])) * self.scale_factor
        pos = self.axis_conversion @ pos

        yaw = radians(row['A'])
        pitch = radians(row['B'])
        roll = radians(row['C'])

        rot = Euler((roll, pitch, yaw), 'XYZ')
        return pos, rot

    # =============================================================
    # 对象列表构建
    # =============================================================
    def create_objects(self):
        objs = []
        for name in self.obj_names:
            if name in bpy.data.objects:
                objs.append(bpy.data.objects[name])
            else:
                print(f"⚠️ 对象未找到: {name}") 
        return objs

    # =============================================================
    # 刚性跟随 main_axis（不依赖父子）
    # =============================================================
    def build_follow_mats(self):
        bpy.context.view_layer.update()
         # 计算 axis 的逆矩阵，用来把物体世界矩阵转换到 axis 坐标系下
        axis_inv = self.axis.matrix_world.inverted()

        follow = self.objects[:]   # 直接复制对象列表
        local = {}

        for o in follow:
            local[o.name] = axis_inv @ o.matrix_world

        return follow, local

    # =============================================================
    # 每帧更新 main_axis / 相机
    # =============================================================
    def update_target(self, tgt_row):
        pos, rot = self.convert_stk_to_blender(tgt_row)
        self.axis.location = pos
        self.axis.rotation_euler = rot

        bpy.context.view_layer.update()

        # 跟随 axis
        for o in self.follow_objects:
            local_mat = self.follow_local_mats[o.name]
            o.matrix_world = self.axis.matrix_world @ local_mat
    # 相机=============================================================
    def update_camera(self, obs_row, tgt_row):
        # 1. 相机位置 = 观测者的 STK 位置（轨道）
        cam_pos, _ = self.convert_stk_to_blender(obs_row)
        self.camera.location = cam_pos

        bpy.context.view_layer.update()

        # 2. 用 track_target（sat_center 或 axis）算 NDC
        target_world = self.track_target.matrix_world.translation
        ndc = world_to_camera_view(self.scene, self.camera, target_world)
        print(f"NDC center: x={ndc.x:.3f}, y={ndc.y:.3f}, z={ndc.z:.3f}")

        # 3. 光照
        self.light.data.type = 'SUN'
        self.light.data.energy = random.uniform(2.4, 2.5)


    # =============================================================
    # bbox
    # =============================================================
    def find_bbox(self, obj_eval, cam_eval):
        deps = bpy.context.evaluated_depsgraph_get()
        mesh = obj_eval.to_mesh(preserve_all_data_layers=True, depsgraph=deps)

        if mesh is None or len(mesh.vertices)==0:
            return None

        xs, ys = [], []
        for v in mesh.vertices:
            w = obj_eval.matrix_world @ v.co #这一步把局部顶点 v.co 变成世界坐标。
            ndc = world_to_camera_view(self.scene, cam_eval, w) #投影到相机坐标
            if ndc.z >= 0:
                xs.append(ndc.x)
                ys.append(1.0 - ndc.y)
        #❗删除临时 mesh 避免泄漏❗
        obj_eval.to_mesh_clear()

        if not xs:
            print("警告: 对象未出现在相机视野内no things in")
            return None

        x1 = float(np.clip(min(xs), 0.0, 1.0))
        x2 = float(np.clip(max(xs), 0.0, 1.0))
        y1 = float(np.clip(min(ys), 0.0, 1.0))
        y2 = float(np.clip(max(ys), 0.0, 1.0))

        if x1==x2 or y1==y2:
            return None

        return (x1,y1),(x2,y2)
    # =============================================================
    # 把 bbox 转成 YOLO 一行：class cx cy w h（全部归一化）
    # 这里 box 是 ((x1,y1),(x2,y2))，已经是 YOLO 坐标系（左上为 0,0）
    # =============================================================
    def yolo_line(self, box, class_id):
        (x1,y1),(x2,y2) = box
        w,h = x2-x1, y2-y1
        cx,cy = x1+w/2, y1+h/2
        return f"{class_id} {cx:.9f} {cy:.9f} {w:.9f} {h:.9f}\n"
    # =============================================================
    # 生成当前帧所有对象的 YOLO 标签文本
    # self.objects 里既可能是单一对象，也可能是对象列表（重复组）
    # 返回一个多行字符串，每行对应一个 bbox
    # =============================================================
    def get_labels(self):
        deps = bpy.context.evaluated_depsgraph_get()
        cam_eval = self.camera.evaluated_get(deps)

        txt=""
        for i,obj in enumerate(self.objects):
            if isinstance(obj,list):
                for o in obj:
                    o_eval = o.evaluated_get(deps)
                    box = self.find_bbox(o_eval, cam_eval)
                    if box:
                        txt+=self.yolo_line(box, i)
            else:
                o_eval = obj.evaluated_get(deps)
                box = self.find_bbox(o_eval, cam_eval)
                if box:
                    txt+=self.yolo_line(box, i)
        return txt

    # =============================================================
    # 渲染函数
    # =============================================================
    def render(self, idx, fidx):
        print(f"\n--- 渲染 {idx} (数据帧 {fidx}) ---")

        self.scene.render.filepath = os.path.join(self.images_filepath,f"{idx:04d}.png")
        bpy.ops.render.render(write_still=True)

        # 写bbox
        with open(os.path.join(self.labels_filepath,f"{idx:04d}.txt"),"w") as f:
            f.write(self.get_labels())

    # =============================================================
    # 主循环
    # =============================================================
    def run(self, num_frames, start_frame=0, step=STEP):
        
        for i in range(num_frames):
            fi = start_frame + i * step

            obs = self.obs_data.iloc[fi]
            tgt = self.tgt_data.iloc[fi]

            self.update_target(tgt)
            self.update_camera(obs, tgt)
            self.render(i+1, fi)
            

# ===================================================================
# 🚀 执行
# ===================================================================
if __name__=="__main__":

    R = RealOrbitFOVRender(output_dir, fov=FOV)
    R.run(NUM, START, step=STEP)
