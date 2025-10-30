import bpy
import numpy as np
from mathutils import Vector
import mathutils

### STEP 1: Find symmetric vertex pairs
C = bpy.context
obj = C.object

# Build KD-tree for vertex lookup
kd_tree = mathutils.kdtree.KDTree(len(obj.data.vertices))
for i, vertex in enumerate(obj.data.vertices):
    kd_tree.insert(vertex.co, i)
kd_tree.balance()

# Search for mirrored vertices across the X-axis
mirror_pairs = []
EPS = 1e-3
for v in obj.data.vertices:
    if v.co[0] < EPS:  # Only check one side to avoid duplicates
        v_mirror = Vector((-v.co[0], v.co[1], v.co[2]))
        mirror_vertex_co, index_mirror, dist = kd_tree.find(v_mirror)
        if dist < EPS and v.index != index_mirror:
            mirror_pairs.append((v.index, index_mirror))
#            obj.data.vertices[index_mirror].select = True  # Select the mirrored vertex
mirror_pairs = np.array(mirror_pairs)

print("Mirror pairs found:", mirror_pairs)

### STEP 2: Select both vertices in each pair if one is selected
for idx_a, idx_b in mirror_pairs:
    if obj.data.vertices[idx_a].select or obj.data.vertices[idx_b].select:
        obj.data.vertices[idx_a].select = True
        obj.data.vertices[idx_b].select = True
