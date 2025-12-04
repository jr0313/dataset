import bpy
import bmesh

# =========================================
# 工具函数：计算物体的 3D 表面积（世界坐标）
# =========================================
def mesh_surface_area(obj: bpy.types.Object) -> float:
    """计算物体的 3D 表面积（转换到世界空间后）"""
    # 从对象生成临时 mesh
    mesh = obj.to_mesh()
    bm = bmesh.new()
    bm.from_mesh(mesh)

    # 应用物体的世界变换（位移/旋转/缩放）
    bm.transform(obj.matrix_world)

    area = sum(f.calc_area() for f in bm.faces)

    bm.free()
    obj.to_mesh_clear()
    return area


# =========================================
# 创建 base_obj 和 part_obj 的布尔交集
# （基于 base_obj 的拷贝，不破坏原模型）
# =========================================
def make_boolean_intersection_object(base_obj, part_obj, name="Intersection_BasePart"):
    """
    返回一个新物体：
    它是 base_obj 和 part_obj 的布尔交集（INTERSECT），
    几何拓扑基于 base_obj 的拷贝。
    """
    # 复制 base_obj 作为布尔载体
    inter_obj = base_obj.copy()
    inter_obj.data = base_obj.data.copy()
    inter_obj.name = name
    bpy.context.collection.objects.link(inter_obj)

    # 确保在 OBJECT 模式
    try:
        bpy.ops.object.mode_set(mode='OBJECT')
    except Exception:
        pass

    # 添加布尔修改器
    bool_mod = inter_obj.modifiers.new(name="Bool_Intersect", type='BOOLEAN')
    bool_mod.operation = 'INTERSECT'
    bool_mod.object = part_obj
    bool_mod.solver = 'FAST'  # 或 'EXACT', 视你的模型复杂度而定

    # 应用修改器
    bpy.ops.object.select_all(action='DESELECT')
    inter_obj.select_set(True)
    bpy.context.view_layer.objects.active = inter_obj
    bpy.ops.object.modifier_apply(modifier=bool_mod.name)

    return inter_obj


# =========================================
# 主函数：计算 “一个物体占另一个物体的表面积比例”
# =========================================
def compute_surface_coverage(base_name: str, part_name: str, delete_intersection: bool = False):
    """
    base_name: 被占用的物体（分母）
    part_name: 占用的物体（分子）
    delete_intersection: 是否在计算完后删除交集物体
    """
    base_obj = bpy.data.objects.get(base_name)
    part_obj = bpy.data.objects.get(part_name)

    if base_obj is None or part_obj is None:
        print(f"❌ 找不到物体：base = '{base_name}', part = '{part_name}'，请检查名字。")
        return

    # 1. 计算 base 的总表面积
    base_area = mesh_surface_area(base_obj)

    # 2. 创建 base ∩ part 的交集物体（基于 base）
    inter_obj = make_boolean_intersection_object(base_obj, part_obj)

    # 3. 计算这个交集物体的表面积
    overlap_area = mesh_surface_area(inter_obj)

    # 4. 占比
    ratio = overlap_area / base_area if base_area > 0 else 0.0

    print("======== 表面积占比（3D） ========")
    print(f"Base 物体名称           : {base_name}")
    print(f"Part 物体名称           : {part_name}")
    print(f"Base 总表面积           : {base_area:.6f}")
    print(f"Base 被 Part 占用的面积 : {overlap_area:.6f}")
    print(f"占比 (over / base)      : {ratio:.6%}")

    # 如果不想看到交集物体，可以删掉
    if delete_intersection:
        bpy.data.objects.remove(inter_obj, do_unlink=True)
    else:
        print(f"✅ 交集物体已保留，名称：{inter_obj.name}")

    return base_area, overlap_area, ratio


# ================================
# 在这里改物体名字，然后运行脚本
# ================================
if __name__ == "__main__":
    # 被占用（分母）的物体，比如地面、墙、平面等
    base_name = "Plane"
    # 占用（分子）的物体，比如立着的 Cube
    part_name = "Cube"

    # delete_intersection=True 代表算完就删掉交集物体
    compute_surface_coverage(base_name, part_name, delete_intersection=False)
