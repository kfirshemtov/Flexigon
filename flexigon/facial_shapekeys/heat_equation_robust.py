import bpy
import bmesh
import numpy as np
import robust_laplacian
import scipy.sparse
import scipy.sparse.linalg

# heat equation parameters
# Time integration
time_steps = 500
delta_t = 0.1

# groupd names parameters
vertex_group_name_from = 'Front_Face'
vertex_group_name_to = 'FeatureWeights'
initial_condition = 'current'
assert initial_condition in ['selected','current'], "initial_condition must be 'selected' or 'current'"

if initial_condition == 'current':
    normalize = False
else:
    normalize = True

## override if needed
normalize = True

# Get the active object
obj = bpy.context.active_object

if obj is None or obj.type != 'MESH':
    raise ValueError("Select a mesh object in the scene.")

mesh = obj.data

# Extract vertices and faces
bm = bmesh.new()
bm.from_mesh(mesh)
bm.verts.ensure_lookup_table()
bm.faces.ensure_lookup_table()

verts = np.array([v.co[:] for v in bm.verts], dtype=np.float64)
faces = np.array([[v.index for v in f.verts] for f in bm.faces if len(f.verts) == 3], dtype=np.int32)

# Compute Laplacian and mass matrix
L, M = robust_laplacian.mesh_laplacian(verts, faces)

# mass matrix TODO: use as is
n = verts.shape[0]
M = scipy.sparse.eye(n)

# Normalize Laplacian using mass matrix (mass-normalized Laplacian)
mass_diag = M.diagonal()
inv_mass_diag = 1.0 / (mass_diag + 1e-10)
M_inv = scipy.sparse.diags(inv_mass_diag)
L_norm = M_inv @ L  # Mass-normalized Laplacian

# Initialize heat source
if initial_condition == 'selected':
    T0 = np.zeros(n)
    source_indices = [v.index for v in obj.data.vertices if v.select]
    for source_index in source_indices:
        if source_index >= n:
            raise IndexError(f"source_index={source_index} exceeds number of vertices={n}")
        T0[source_index] = 1.0
elif initial_condition == 'current':
    # Use existing vertex group weights as initial heat
    vertex_group = obj.vertex_groups.get(vertex_group_name_from)
    if not vertex_group:
        raise ValueError(f"Vertex group '{vertex_group_name}' not found. Please create and assign initial weights.")

T0 = np.zeros(n)
for i, v in enumerate(obj.data.vertices):
    for g in v.groups:
        if g.group == vertex_group.index:
            T0[i] = g.weight


T = T0.copy()

A = scipy.sparse.identity(n) + delta_t * L_norm
solver = scipy.sparse.linalg.factorized(A)

for _ in range(time_steps):
    T = solver(T)

# Normalize temperature values
if normalize:
    T_vis = (T - T.min()) / (T.max() - T.min() + 1e-20)
else:
    T_vis = T

# Assign vertex group weights
vertex_group = obj.vertex_groups.get(vertex_group_name_to)
if not vertex_group:
    vertex_group = obj.vertex_groups.new(name=vertex_group_name_to)

if len(T_vis) != len(obj.data.vertices):
    raise ValueError("Mismatch between vertex count and heat values.")

for i, value in enumerate(T_vis):
    vertex_group.add([i], value, 'REPLACE')

print("Assigned normalized heat weights to vertex group.")
