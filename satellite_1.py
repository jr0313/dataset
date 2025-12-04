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
OBJ_NAMES = ['panel1', 'panel2', 'panel3', 'panel4','panel5','panel6','satellite1','hole']  # 若无需求可以留空

# STK CSV 路径
STK_PATHS = {
    "OBS_POS": r"E:\jr\StkData\1.MSX-J2000 Position Velocity.csv",
    "OBS_ATT": r"E:\jr\StkData\1.MSX-Euler Angles.csv",
    "TGT_POS": r"E:\jr\StkData\1.CloudSat-J2000 Position Velocity.csv",
    "TGT_ATT": r"E:\jr\StkData\1.CloudSat-Euler Angles.csv",
}

# 输出目录
output_dir = r"E:\jr\SpaceTarget\1\4.output"
os.makedirs(output_dir, exist_ok=True)  # 防止目录不存在
OUTPUT_PATHS = {
    "IMAGES": os.path.join(output_dir, "Data_Real"),  
    "LABELS": os.path.join(output_dir, "Labels"),
}

# === 新增：面积比例 CSV 路径 ===
CSV_PATH = os.path.join(output_dir, "area_ratio2D.csv")

# === 新增：计算帧内 部件/帆板 像素面积比例，并写入 CSV ===
PANEL_NAME = "panel3"   # 帆板物体名（分母）
PART_NAME  = "hole"     # 部件物体名（分子）
# 渲染参数
FOV = 40
NUM = 50
START = 0
STEP = 40

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
        self.images_filepath = OUTPUT_PATHS["IMAGES"]
        self.labels_filepath = OUTPUT_PATHS["LABELS"]
        os.makedirs(self.images_filepath, exist_ok=True)
        os.makedirs(self.labels_filepath, exist_ok=True)

        # 渲染设置不变
        self.scene.render.engine = 'CYCLES'  # 启用光线追踪渲染引擎
        self.scene.render.image_settings.file_format = 'PNG'
        self.scene.cycles.samples = 128
        self.scene.cycles.device = 'GPU'
        self.scene.cycles.tile_size = 256
        self.scene.render.image_settings.color_mode = 'RGBA'  #? 启用透明背景

        # 分辨率
        self.scene.render.resolution_x = 1280
        self.scene.render.resolution_y = 1280
        self.scene.render.resolution_percentage = 100

        # 加载轨道与姿态
        self.load_stk_data()

        # 坐标系转化
        self.scale_factor = 0.001  # 将STK的公里单位转换为Blender的米单位
        self.axis_conversion = Matrix.Rotation(radians(90), 4, 'X')

        # FOV
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
        
        df = pd.read_csv(csv_path)
        df.columns = [c.split()[0] for c in df.columns]
        df['Time'] = pd.to_datetime(df['Time'], format='mixed', dayfirst=False, errors='coerce')
        if df.empty:
            raise ValueError(f"CSV文件为空或格式错误: {csv_path}")
        print(f"✅ 位置数据加载成功: {len(df)} 行")
        return df[['Time', 'x', 'y', 'z', 'vx', 'vy', 'vz']]

    def read_stk_euler(self, csv_path):
        """读取STK欧拉角（极简版）"""
        df = pd.read_csv(csv_path)
        df.columns = [c.split()[0] for c in df.columns]
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
        obs_pos_path = STK_PATHS["OBS_POS"]
        obs_att_path = STK_PATHS["OBS_ATT"]
        tgt_pos_path = STK_PATHS["TGT_POS"]
        tgt_att_path = STK_PATHS["TGT_ATT"]

        obs_pos = self.read_stk_position(obs_pos_path)
        tgt_pos = self.read_stk_position(tgt_pos_path)

        obs_att = self.read_stk_euler(obs_att_path)
        tgt_att = self.read_stk_euler(tgt_att_path)

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

    # 相机==========================================================
    def update_camera(self, obs_row, tgt_row):
        cam_pos, _ = self.convert_stk_to_blender(obs_row)
        self.camera.location = cam_pos

        bpy.context.view_layer.update()

        target_world = self.track_target.matrix_world.translation
        ndc = world_to_camera_view(self.scene, self.camera, target_world)
        print(f"NDC center: x={ndc.x:.3f}, y={ndc.y:.3f}, z={ndc.z:.3f}")

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
            w = obj_eval.matrix_world @ v.co
            ndc = world_to_camera_view(self.scene, cam_eval, w)
            if ndc.z >= 0:
                xs.append(ndc.x)
                ys.append(1.0 - ndc.y)
        obj_eval.to_mesh_clear()

        if not xs:
            print("警告: 对象未出现在相机视野内")
            return None

        x1 = float(np.clip(min(xs), 0.0, 1.0))
        x2 = float(np.clip(max(xs), 0.0, 1.0))
        y1 = float(np.clip(min(ys), 0.0, 1.0))
        y2 = float(np.clip(max(ys), 0.0, 1.0))

        if x1==x2 or y1==y2:
            return None

        return (x1,y1),(x2,y2)

    # =============================================================
    # === 新增：渲染单个物体为透明背景，统计像素面积 ===
    # =============================================================
    # =============================================================
    # 渲染单个物体为透明背景 PNG，统计 alpha>0 的像素数
    # =============================================================
    def render_object_mask_pixels(self, obj_name: str) -> int:
        """
        在当前相机姿态、当前帧下：
        只渲染 obj_name 这个物体到一个临时 PNG 文件，
        再读取该 PNG，统计 alpha>0 的像素数，作为该物体在图像中的投影面积（像素）。
        """
        scene = self.scene
        obj = bpy.data.objects.get(obj_name)
        if obj is None:
            print(f"⚠ render_object_mask_pixels: 找不到物体 {obj_name}")
            return 0

        width  = scene.render.resolution_x
        height = scene.render.resolution_y
        total_pixels = width * height

        # 备份状态
        orig_hide_render = {o.name: o.hide_render for o in scene.objects}
        orig_film_transparent = scene.render.film_transparent
        orig_filepath = scene.render.filepath
        orig_material = obj.active_material  # 备份原材质

        tmp_path = os.path.join(self.images_filepath,'white', f"mask_tmp_{obj_name}.png")

        mask_mat = None  # 先设为 None，避免 finally 未定义

        try:
            # === 1. 创建纯白不透明材质并替换原材质 ===
            mask_mat = bpy.data.materials.new(name="__mask_mat_temp")
            mask_mat.use_nodes = True

            # 清空默认节点，自己搭建
            nt = mask_mat.node_tree
            for n in list(nt.nodes):
                nt.nodes.remove(n)

            bsdf = nt.nodes.new('ShaderNodeBsdfPrincipled')
            output = nt.nodes.new('ShaderNodeOutputMaterial')
            nt.links.new(bsdf.outputs['BSDF'], output.inputs['Surface'])

            bsdf.inputs["Base Color"].default_value = (1.0, 1.0, 1.0, 1.0)
            bsdf.inputs["Alpha"].default_value = 1.0
            mask_mat.blend_method = 'OPAQUE'

            obj.active_material = mask_mat

            # === 2. 设置透明背景 & RGBA ===
            scene.render.film_transparent = True
            scene.render.image_settings.color_mode = 'RGBA'

            # === 3. 控制可见性 ===
            for o in scene.objects:
                if o.type in {'CAMERA', 'LIGHT'}:
                    o.hide_render = False
                elif o == obj:
                    o.hide_render = False
                elif o.type == 'MESH':
                    o.hide_render = True  # 只隐藏其他 mesh
                else:
                    o.hide_render = False

            # === 4. 输出 PNG ===
            scene.render.filepath = tmp_path
            bpy.ops.render.render(write_still=True)

            # === 5. 读取 PNG 的 alpha ===
            img = bpy.data.images.load(tmp_path, check_existing=True)
            img.reload()  
            pixels = np.array(img.pixels[:])  # [R,G,B,A, R,G,B,A, ...]
            if pixels.size == 0:
                print(f"⚠ render_object_mask_pixels: 图像像素为空, obj={obj_name}")
                return 0

            alpha = pixels[3::4]
            max_a = float(alpha.max())
            obj_pixels = int(np.count_nonzero(alpha > 1e-4))

            print(f"🔍 调试: {obj_name} 的 alpha 最大值 = {max_a:.4f}")
            print(f"🔹 物体 {obj_name} 像素 = {obj_pixels} / {total_pixels}")

            return obj_pixels

        finally:
            # === 恢复 hide_render 状态 ===
            for name, val in orig_hide_render.items():
                if name in scene.objects:
                    scene.objects[name].hide_render = val

            scene.render.film_transparent = orig_film_transparent
            scene.render.filepath = orig_filepath

            # === 恢复原材质 ===
            obj.active_material = orig_material

            # 删除临时材质
            if mask_mat is not None and mask_mat.name in bpy.data.materials:
                bpy.data.materials.remove(mask_mat)

            # 可以视情况删除临时图片文件（如果你不想留在磁盘上）
            # if os.path.exists(tmp_path):
            #     os.remove(tmp_path)
    # =============================================================
    # === 新增：计算 帆板/部件 像素面积比例，并返回数值 ===
    # =============================================================
    def compute_panel_part_area_ratio(self, panel_name: str, part_name: str):
        panel_pixels = self.render_object_mask_pixels(panel_name)
        part_pixels  = self.render_object_mask_pixels(part_name)

        if panel_pixels <= 0:
            print(f"⚠ 帆板 {panel_name} 像素数为 0，无法计算比例")
            return None

        ratio = part_pixels / panel_pixels
        print("\n=========== 当前帧 2D 投影面积比例 (像素) ===========")
        print(f"帆板 (分母) : {panel_name} 像素 = {panel_pixels}")
        print(f"部件 (分子) : {part_name} 像素 = {part_pixels}")
        print(f"部件占帆板面积比例 = {ratio:.6f}")
        print("===================================================\n")

        return panel_pixels, part_pixels, ratio

    # =============================================================
    # 把 bbox 转成 YOLO 一行
    # =============================================================
    def yolo_line(self, box, class_id):
        (x1,y1),(x2,y2) = box
        w,h = x2-x1, y2-y1
        cx,cy = x1+w/2, y1+h/2
        return f"{class_id} {cx:.9f} {cy:.9f} {w:.9f} {h:.9f}\n"

    # =============================================================
    # 生成当前帧所有对象的 YOLO 标签文本
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
    def render(self, idx, fidx, cal = True):
        print(f"\n--- 渲染 {idx} (数据帧 {fidx}) ---")

        # 正常渲染图像
        self.scene.render.filepath = os.path.join(self.images_filepath,f"{idx:04d}.png")
        bpy.ops.render.render(write_still=True)

        # 写bbox标签
        with open(os.path.join(self.labels_filepath,f"{idx:04d}.txt"),"w") as f:
            f.write(self.get_labels())

        # 计算帆板和部件的像素面积比例，并写入 CSV
        # 注意：这里假设帆板和部件在每帧都存在
        result = self.compute_panel_part_area_ratio(PANEL_NAME, PART_NAME)
        if result is not None:
            panel_px, part_px, ratio = result

            # 追加写入 CSV
            # 第一帧时，如果文件为空/不存在，则写表头
            write_header = (idx == 1 and (not os.path.exists(CSV_PATH) or os.path.getsize(CSV_PATH) == 0))
            with open(CSV_PATH, "a", encoding="utf-8") as f:
                if write_header:
                    f.write("frame,panel_pixels,part_pixels,ratio\n")
                f.write(f"{idx},{panel_px},{part_px},{ratio}\n")

    # =============================================================
    # 主循环
    # =============================================================
    def run(self, num_frames, start_frame=0, step=STEP, cal = True):
        for i in range(num_frames):
            fi = start_frame + i * step

            obs = self.obs_data.iloc[fi]
            tgt = self.tgt_data.iloc[fi]

            self.update_target(tgt)
            self.update_camera(obs, tgt)
            self.render(i+1, fi, cal)
            

# ===================================================================
# 🚀 执行
# ===================================================================
if __name__=="__main__":

    R = RealOrbitFOVRender(output_dir, fov=FOV)
    R.run(NUM, START, step=STEP, cal=True)