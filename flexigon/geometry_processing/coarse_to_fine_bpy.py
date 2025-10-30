import os
import bpy
import torch
from pytorch3d.io import load_obj , save_obj
from mathutils.geometry import barycentric_transform
from mathutils.interpolate import poly_3d_calc
import trimesh
from scipy.spatial import KDTree
import numpy as np
import gdist

EPS = 1e-8

### pre
# decimate mesh using blender

def delete_all_objects():
    # Deselect all objects first
    bpy.ops.object.select_all(action='DESELECT')

    # Select only mesh objects
    for obj in bpy.context.scene.objects:
        if obj.type == 'MESH':
            obj.select_set(True)

    # Make one of the selected objects active (required for some ops)
    if bpy.context.selected_objects:
        bpy.context.view_layer.objects.active = bpy.context.selected_objects[0]

    # Now delete selected objects
    bpy.ops.object.delete()
    return

def create_low_mesh(obj_to_decimate , loc_to_save,ratio , high_level, low_level):
    """
    Loads a high-resolution mesh, applies decimation using the specified ratio,
    and saves both the original (if it's the first level) and decimated mesh as .obj files.
    Cleans the Blender scene before loading and exporting.

    Parameters:
        obj_to_decimate (str): Path to the high-resolution .obj file.
        loc_to_save (str): Directory to save the output .obj files.
        ratio (float): Decimation ratio (0 < ratio <= 1).
        high_level (int): Identifier for the original high-resolution mesh level.
        low_level (int): Identifier for the new low-resolution mesh level.
    """

    high_level = 'level' + str(high_level)
    low_level = 'level' + str(low_level)

    # clean scene: remove all objects
    delete_all_objects()

    # load mesh and decimate
    bpy.ops.import_scene.obj(filepath=obj_to_decimate)
    obj_hr_bpy = [ob for ob in bpy.context.scene.objects if ob.type in ('MESH')][0]
    obj_hr_bpy.select_set(True)
    bpy.context.view_layer.objects.active = obj_hr_bpy

    # save the high level mesh for the first (next levels are already saved from the previous decimation)
    if high_level == 'level0':
        export_name = os.path.join(loc_to_save, high_level + '.obj')
        os.makedirs(loc_to_save, exist_ok=True)
        bpy.ops.export_scene.obj(filepath=export_name, check_existing=False, filter_glob="*.obj;*.mtl",
                                 use_selection=True, use_animation=False, use_mesh_modifiers=True, use_edges=True,
                                 use_smooth_groups=False, use_smooth_groups_bitflags=False, use_normals=True,
                                 use_uvs=True, use_materials=False, use_triangles=True, use_nurbs=False,
                                 use_vertex_groups=False, use_blen_objects=True, group_by_object=False,
                                 group_by_material=False, keep_vertex_order=True, global_scale=1, path_mode='AUTO',
                                 axis_forward='-Z', axis_up='Y')
    bpy.ops.object.modifier_add(type='DECIMATE')
    bpy.context.object.modifiers["Decimate"].use_symmetry = True
    bpy.context.object.modifiers["Decimate"].ratio = ratio
    bpy.ops.object.modifier_apply(modifier="Decimate")
    # export the low level mesh
    export_name = os.path.join(loc_to_save, low_level + '.obj')
    bpy.ops.export_scene.obj(filepath=export_name, check_existing=False, filter_glob="*.obj;*.mtl",
                             use_selection=True, use_animation=False, use_mesh_modifiers=True, use_edges=True,
                             use_smooth_groups=False, use_smooth_groups_bitflags=False, use_normals=True,
                             use_uvs=True, use_materials=False, use_triangles=True, use_nurbs=False,
                             use_vertex_groups=False, use_blen_objects=True, group_by_object=False,
                             group_by_material=False, keep_vertex_order=True, global_scale=1, path_mode='AUTO',
                             axis_forward='-Z', axis_up='Y')


def calculate_hr_to_lr_from_uv(obj_to_decimate , loc_to_save , high_level, low_level):
    """
    Computes a sparse transformation matrix from a low-resolution mesh to a high-resolution mesh
    using UV-space barycentric mapping. It builds auxiliary UV-mapped meshes,
    finds corresponding faces, and calculates interpolation weights.

    Parameters:
        high_level (str): Identifier for the high-resolution mesh level.
        low_level (str): Identifier for the low-resolution mesh level.

    Returns:
        torch.sparse.FloatTensor: Sparse transformation matrix (high_res_vertices x low_res_vertices).
        np.ndarray: Face indices of the high-resolution mesh.
    """
    high_level = 'level' + str(high_level)
    low_level = 'level' + str(low_level)

    # prepare virtual uv mesh for the original mesh and decimated mesh
    # high level mesh
    obj_high_level = load_obj(os.path.join(loc_to_save, high_level + '.obj'))
    obj = obj_high_level
    save_obj(os.path.join(loc_to_save,high_level + '_uv.obj'),verts=torch.cat((obj[2][1],torch.zeros(len(obj[2][1]),1)),dim=1),faces=obj[1][2])
    # low level mesh
    obj_low_level = load_obj(os.path.join(loc_to_save,low_level + '.obj'))
    obj = obj_low_level
    save_obj(os.path.join(loc_to_save,low_level + '_uv.obj'),verts=torch.cat((obj[2][1],torch.zeros(len(obj[2][1]),1)),dim=1),faces=obj[1][2])
    ## load
    # clean scene: remove all objects
    objs_to_delete = [ob for ob in bpy.context.scene.objects if ob.type in ('MESH')]
    bpy.ops.object.delete({"selected_objects": objs_to_delete})
    bpy.ops.import_scene.obj(filepath=obj_to_decimate)
    obj_hr_bpy = [ob for ob in bpy.context.scene.objects if ob.type in ('MESH')][0]
    obj_hr_bpy.select_set(True)

    # load uv virtual mesh
    obj_path = os.path.join(loc_to_save,low_level + '_uv.obj')
    bpy.ops.import_scene.obj(filepath=obj_path)
    obj_uv = [ob for ob in bpy.context.scene.objects if ob.type in ('MESH') and 'level' in ob.name][0]
    obj_uv.select_set(True)

    low2high_idx_x = []  # used for building sparse matrix
    low2high_idx_y = []  # used for building sparse matrix
    value = []  # used for building sparse matrix

    obj_hr = trimesh.load(os.path.join(loc_to_save , high_level + '.obj') , merge_norm=True, merge_tex=True)
    obj_uv_hr = trimesh.load(os.path.join(loc_to_save , high_level + '_uv.obj') , merge_norm=True, merge_tex=True)
    obj_lr = trimesh.load(os.path.join(loc_to_save , low_level + '.obj') , merge_norm=True, merge_tex=True)
    obj_uv_lr = trimesh.load(os.path.join(loc_to_save , low_level + '_uv.obj') , merge_norm=True, merge_tex=True)
    for v_idx,vert in enumerate(obj_hr.vertices):
        flag_success, close_point, _, face_idx = obj_hr_bpy.closest_point_on_mesh(vert)
        assert flag_success == True, 'close point algorithm has failed'
        vertices_hr_mesh = obj_hr.vertices[obj_hr.faces[face_idx]]
        vertices_hr_uv = obj_uv_hr.vertices[obj_uv_hr.faces[face_idx]]

        # calculate the uv_point
        tri_a1 = vertices_hr_mesh[0]
        tri_a2 = vertices_hr_mesh[1]
        tri_a3 = vertices_hr_mesh[2]
        tri_b1 = vertices_hr_uv[0]
        tri_b2 = vertices_hr_uv[1]
        tri_b3 = vertices_hr_uv[2]
        uv_point = barycentric_transform(close_point, tri_a1, tri_a2, tri_a3, tri_b1, tri_b2, tri_b3)

        # get the face of the low level mesh
        flag_success, close_point, _, face_idx = obj_uv.closest_point_on_mesh(uv_point)
        vertices_uv_idx_low_level = obj_uv_lr.vertices[obj_uv_lr.faces[face_idx]]
        barycentric_co = poly_3d_calc(vertices_uv_idx_low_level, close_point)

        # add to sparse matrix
        neighbors_idx = obj_lr.faces[face_idx]
        for i in range(3):
            low2high_idx_x.append(torch.tensor([v_idx]))
            low2high_idx_y.append(torch.tensor([neighbors_idx[i]]))
            value.append(torch.tensor(barycentric_co[i]))

    low2high_idx = torch.cat((torch.stack(low2high_idx_x), torch.stack(low2high_idx_y)), dim=1).t()
    value = torch.stack(value)
    low2high_transform = torch.sparse.FloatTensor(low2high_idx, value,
                                                  (len(obj_hr.vertices), len(obj_lr.vertices)))

    return low2high_transform

def calculate_hr_to_lr_from_distance(loc_to_save , high_level, low_level,k=3,p=2):
    high_level = 'level' + str(high_level)
    low_level = 'level' + str(low_level)

    # read lr and hr objects
    obj_hr = trimesh.load(os.path.join(loc_to_save , high_level + '.obj') , merge_norm=True, merge_tex=True)
    obj_lr = trimesh.load(os.path.join(loc_to_save , low_level + '.obj') , merge_norm=True, merge_tex=True)

    # create KD Tree to find efficiently the closest points
    ## TODO: change to geodesic distance
    tree_lr = KDTree(obj_lr.vertices)
    dists, idxs = tree_lr.query(obj_hr.vertices, k=k)


    # Handle case where k=1 to ensure 2D shape
    if k == 1:
        dists = dists[:, np.newaxis]
        idxs = idxs[:, np.newaxis]

    weights = 1.0 / (np.power(dists, p) + EPS)
    weights_sum = np.sum(weights, axis=1, keepdims=True)
    weights = weights / weights_sum  # Normalize to sum to 1

    # Now build sparse matrix
    hr_n = obj_hr.vertices.shape[0]
    lr_n = obj_lr.vertices.shape[0]

    row_idx = []
    col_idx = []
    values = []

    for i in range(hr_n):
        for j in range(k):
            row_idx.append(i)
            col_idx.append(idxs[i, j])
            values.append(weights[i, j])

    indices = torch.LongTensor([row_idx, col_idx])
    values = torch.FloatTensor(values)
    transform = torch.sparse.FloatTensor(indices, values, torch.Size([hr_n, lr_n]))

    return transform


def calculate_hr_to_lr_from_geodesic(obj_to_decimate , loc_to_save , high_level, low_level,k=8,p=1):
    high_level = 'level' + str(high_level)
    low_level = 'level' + str(low_level)

    # read lr and hr objects
    obj_hr = trimesh.load(os.path.join(loc_to_save , high_level + '.obj') , merge_norm=True, merge_tex=True)
    obj_lr = trimesh.load(os.path.join(loc_to_save , low_level + '.obj') , merge_norm=True, merge_tex=True)

    lr_v = obj_lr.vertices.astype(np.float64)
    lr_f = obj_lr.faces.astype(np.int32)
    hr_v = obj_hr.vertices.astype(np.float64)
    hr_n = len(hr_v)
    lr_n = len(lr_v)

    row_idx = []
    col_idx = []
    values = []

    # Build face tree for projection
    face_tree = trimesh.proximity.ProximityQuery(obj_lr)

    for i, pt in enumerate(hr_v):
        # Step 1: project to surface
        closest_pts, dists, face_idxs = face_tree.on_surface([pt])
        closest_pt = closest_pts[0]
        face_idx = face_idxs[0]

        face = lr_f[face_idx]
        verts = lr_v[face]

        # Step 2: bary coords on the triangle
        bc = trimesh.triangles.points_to_barycentric(verts[None], [closest_pt])[0]

        # Step 3: compute geodesic distances from each vertex in the triangle
        d_all = []
        for j in range(3):
            dists = gdist.compute_gdist(lr_v, lr_f, source_indices=np.array([face[j]], dtype=np.int32))
            d_all.append(dists)

        # Step 4: weighted geodesic distances using barycentric coords
        d_combined = bc[0]*d_all[0] + bc[1]*d_all[1] + bc[2]*d_all[2]

        # Step 5: choose k nearest neighbors and compute weights
        nearest_idxs = np.argsort(d_combined)[:k]
        nearest_dists = d_combined[nearest_idxs]
        weights = 1.0 / (np.power(nearest_dists, p) + EPS)
        weights /= weights.sum()

        for j in range(k):
            row_idx.append(i)
            col_idx.append(nearest_idxs[j])
            values.append(weights[j])

    indices = torch.LongTensor([row_idx, col_idx])
    values = torch.FloatTensor(values)
    transform = torch.sparse.FloatTensor(indices, values, torch.Size([hr_n, lr_n]))

    return transform


def decimate_mesh_by_geodesic(obj_to_decimate , loc_to_save,ratio=0.66 , high_level=0, low_level=1):
    create_low_mesh(obj_to_decimate , loc_to_save,ratio , high_level, low_level)
    return calculate_hr_to_lr_from_geodesic(obj_to_decimate , loc_to_save , high_level, low_level)


def decimate_mesh_by_uv(obj_to_decimate , loc_to_save,ratio=0.66 , high_level=0, low_level=1):
    """
    Decimates a high-resolution mesh and computes the UV-based sparse transformation matrix
    from the resulting low-resolution mesh back to the high-resolution mesh.

    Parameters:
        obj_to_decimate (str): Path to the high-resolution .obj file.
        loc_to_save (str): Directory to save intermediate and final .obj files.
        ratio (float): Decimation ratio (default: 0.66).
        high_level (int): Identifier for the original high-resolution mesh level.
        low_level (int): Identifier for the new low-resolution mesh level.

    Returns:
        torch.sparse.FloatTensor: Sparse transformation matrix (high_res_vertices x low_res_vertices).
        np.ndarray: Face indices of the high-resolution mesh.
    """

    create_low_mesh(obj_to_decimate , loc_to_save,ratio , high_level, low_level)
    return calculate_hr_to_lr_from_uv(obj_to_decimate , loc_to_save , high_level, low_level)
