import bpy
import os
from natsort import natsorted
import numpy as np
from sklearn.decomposition import PCA
from scipy.spatial.distance import cdist

class CreateBlenderShapekeys:
    def __init__(self, input_folder, output_folder=None):
        '''
            Create Blender shape keys and animations from a set of OBJ meshes.

    Args:
        input_folder (str): Path to folder containing OBJ files.
        output_folder (str, optional): Path to folder where Blender files will be saved.
            Defaults to the same folder as input_folder.
        '''
        self.input_folder = input_folder
        self.output_folder = output_folder if output_folder else input_folder
        self.files = natsorted([f for f in os.listdir(input_folder) if f.endswith(".obj")])
        if not self.files:
            raise ValueError("No OBJ files found in folder!")
        self.mesh_data = {}  # store vertices and faces
        self.X_flat = None
        self.X_reduced = None
        self.pca = None

    def load_meshes(self, load_all=True):
        """Load meshes from OBJ files and store vertices and faces"""
        for f in self.files if load_all else self.files[:5]:
            if f in self.mesh_data:
                continue
            path = os.path.join(self.input_folder, f)
            bpy.ops.import_scene.obj(filepath=path,
                                     axis_forward='-Z',
                                     axis_up='Y',
                                     )
            obj = bpy.context.selected_objects[0]
            bpy.ops.object.shade_smooth()

            # Store vertices
            verts = np.array([v.co for v in obj.data.vertices])
            # Store faces as list of indices
            faces = [list(p.vertices) for p in obj.data.polygons]

            # Store UV coordinates (if available)
            uv_layer = None
            if obj.data.uv_layers.active:
                uv_layer = np.array([loop.uv for loop in obj.data.uv_layers.active.data])

            self.mesh_data[f] = {
                "verts": verts,
                "faces": faces,
                "uvs": uv_layer,
            }
            bpy.data.objects.remove(obj, do_unlink=True)

        # Flatten vertices for PCA
        self.X_flat = np.array([v["verts"].flatten() for v in self.mesh_data.values()])

    def compute_pca(self, n_components=6):
        """Compute PCA from mesh data"""
        n_samples, n_features = self.X_flat.shape
        n_components = min(n_components, n_samples, n_features)
        self.pca = PCA(n_components=n_components)
        self.X_reduced = self.pca.fit_transform(self.X_flat)

    def select_topk(self, K=5):
        """Select the top K meshes that best represent the variance in the dataset.

        The method works as follows:
        1. Compute pairwise Euclidean distances between all reduced PCA embeddings.
        2. Start by selecting the mesh farthest from the mean (most "extreme" in the dataset).
        3. Iteratively select the mesh that maximizes the minimum distance to already selected meshes,
           ensuring a diverse set covering different modes of variation.
        4. Return the corresponding file names of the selected meshes.
        """
        if self.X_reduced is None:
            raise RuntimeError("PCA not computed yet")
        D = cdist(self.X_reduced, self.X_reduced, metric='euclidean')
        N = len(self.files)
        remaining = list(range(N))
        selected_idx = []
        mean_point = np.mean(self.X_reduced, axis=0, keepdims=True)
        first_idx = np.argmax(cdist(self.X_reduced, mean_point).flatten())
        selected_idx.append(first_idx)
        remaining.remove(first_idx)
        for _ in range(K-1):
            min_dists = np.min(D[remaining][:, selected_idx], axis=1)
            next_idx = remaining[np.argmax(min_dists)]
            selected_idx.append(next_idx)
            remaining.remove(next_idx)
        return [self.files[i] for i in selected_idx]

    def create_blender_file(self, files_subset, filename, start_frame=0, end_frame=250, incremental=False):
        """
        Create a Blender file with shape keys and animation from a subset of meshes.

        Args:
            files_subset (list): List of mesh filenames to use.
            filename (str): Name of the Blender file to save.
            start_frame (int): Start frame for animation.
            end_frame (int): End frame for animation.
            incremental (bool): If True, each shape key is relative to previous mesh.
        """
        # Clear existing objects
        bpy.ops.object.select_all(action='SELECT')
        bpy.ops.object.delete(use_global=False)

        # Base mesh
        base_file = files_subset[0]
        base_data = self.mesh_data[base_file]
        base_mesh = bpy.data.meshes.new("BaseMesh")
        base_mesh.from_pydata(base_data["verts"].tolist(), [], base_data["faces"])
        base_mesh.update()
        base_obj = bpy.data.objects.new("BaseObject", base_mesh)
        bpy.context.collection.objects.link(base_obj)
        bpy.context.view_layer.objects.active = base_obj
        base_obj.select_set(True)
        bpy.ops.object.shade_smooth()
        base_obj.rotation_euler[0] = np.pi / 2

        # Recreate UV map if available
        if "uvs" in base_data and base_data["uvs"] is not None:
            uv_layer = base_mesh.uv_layers.new(name="UVMap")
            for li, loop in enumerate(base_mesh.loops):
                uv_layer.data[li].uv = base_data["uvs"][li]

        # Reassign materials if available
        if "materials" in base_data and base_data["materials"]:
            for mat in base_data["materials"]:
                base_mesh.materials.append(mat)

        # Add Basis shape key
        base_obj.shape_key_add(name="Basis")
        prev_key = base_obj.data.shape_keys.key_blocks["Basis"]

        # Add shape keys
        for f in files_subset[1:]:
            data = self.mesh_data[f]
            key_name = os.path.splitext(f)[0]
            if incremental:
                key_name = 'delta_' + key_name
            key = base_obj.shape_key_add(name=key_name)
            # assign vertex positions
            for i, v in enumerate(data["verts"]):
                key.data[i].co = v
            if incremental:
                # make relative to previous key
                key.relative_key = prev_key
                # set slider limits
                key.slider_min = -3
                key.slider_max = 3
            prev_key = key

        # Animate shape keys
        shape_keys = base_obj.data.shape_keys.key_blocks
        n_keys = len(shape_keys)
        if n_keys > 1:
            frames_per_key = (end_frame - start_frame) / (n_keys - 1)
            for i, key in enumerate(shape_keys):
                frame = start_frame + i * frames_per_key
                if incremental:
                    # default sequential animation from Basis
                    for j,key_in in enumerate(shape_keys):
                        if j <= i:
                            key_in.value = 1.0
                        else:
                            key_in.value = 0.0
                        key_in.keyframe_insert(data_path="value", frame=frame)
                    key.value = 1.0
                    key.keyframe_insert(data_path="value", frame=frame)
                elif not incremental:
                    # default sequential animation from Basis
                    for key_in in shape_keys:
                        key_in.value = 0.0
                        key_in.keyframe_insert(data_path="value", frame=frame)
                    key.value = 1.0
                    key.keyframe_insert(data_path="value", frame=frame)

        # Save Blender file
        output_path = os.path.join(self.output_folder, filename)
        bpy.ops.wm.save_mainfile(filepath=output_path)
        print(f"Saved Blender file: {output_path}")

    def __call__(self, start_frame=0,end_frame=250,topk=5,incremental='both'):
        # Load all meshes once
        self.load_meshes()

        # Compute PCA
        self.compute_pca()

        # Create Blender file with all meshes
        self.create_blender_file(self.files,filename="mesh_evolution_all.blend",start_frame=start_frame,end_frame=end_frame)

        # Select top K most different meshes
        top_files = self.select_topk(K=topk)
        top_files_sorted = [f for f in self.files if f in top_files]
        if incremental == 'both':
            self.create_blender_file(top_files_sorted, f"mesh_top{topk}.blend", incremental=False)
            self.create_blender_file(top_files_sorted, f"mesh_top{topk}_incremental.blend", incremental=True)
        elif incremental == True:
            self.create_blender_file(top_files_sorted, f"mesh_top{topk}_incremental.blend", incremental=True)
        else:
            self.create_blender_file(top_files_sorted, f"mesh_top{topk}.blend", incremental=False)

        return