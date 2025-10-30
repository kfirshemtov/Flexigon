import argparse
import os
import shutil
import torch
import flexigon
from flexigon.geometry_processing.geometry_utils import remove_duplicates, compute_face_normals, compute_vertex_normals
import torchvision.transforms as transforms
import numpy as np
from flexigon.magic_shape_keys.magic_shapekeys_template import MagicShapeKeys
import matplotlib.pyplot as plt
from flexigon.geometry_processing.coarse_to_fine_bpy import decimate_mesh_by_uv
from flexigon.geometry_processing.sobolev_utils import compute_matrix
from flexigon.geometry_processing.parameterize import to_differential, from_differential
from flexigon.optimizers.optimizer_utils import initialize_optimizer
from flexigon.geometry_processing.replace_vertices import replace_vertices
from flexigon.facial_shapekeys.create_shapekeys_from_flexigon import CreateBlenderShapekeys
from functools import reduce
import trimesh
from PIL import Image
import cv2
from tqdm import tqdm


class MagicShapeKeysMultiRes(MagicShapeKeys):
    def __init__(self,config_path,scene_path,**kwargs):
        super().__init__(config_path,scene_path,**kwargs)  # Call parent class constructor
        self.package_path = os.path.dirname(flexigon.__path__[0])
        self.init_default_values(self.cfg)
        self.init_multires()
        self.load_multires_mesh()

    def init_default_values(self,config):
        super().init_default_values(config)
        config.setdefault("debug_mode", False)
        config.setdefault("res_level", 1)
        config.setdefault("decimate_ratio")
        config.setdefault("override_prev_decimate", True)
        config.setdefault("relative_loss", False)
        config.setdefault("lambda_relative_loss", 0.1)
        config.setdefault("lambda_negative", 1)
        config.setdefault("reinit_opt_every", 300)
        config.setdefault("deform_length_t_ratio_h", 0.002)
        config.setdefault("mesh_scale_factor", 1.0)
        config.setdefault("mesh_std_norm_scale",  [0.05, 0.1, 0.06])
        config.setdefault("mesh_offset", [0.0, 0.0, 0.0])
        config.setdefault("mesh_name", 'mesh')
        return

    def load_multires_mesh(self):
        # load the HR mesh
        self.mesh_hr = trimesh.load(os.path.join(self.multires_path,'level0.obj'),merge_norm=True, merge_tex=True)
        self.vertices_hr = self.mesh_hr.vertices
        self.triangles_hr = self.mesh_hr.faces
        self.vertices_hr = torch.tensor(self.vertices_hr, dtype=torch.float32, device=self.device)
        self.triangles_hr = torch.tensor(self.triangles_hr, dtype=torch.int32, device=self.device)
        _,_,self.duplicate_idx_hr = remove_duplicates(self.vertices_hr,self.triangles_hr)

        # read texture info
        self.uvs = torch.tensor(self.mesh_hr.visual.uv, dtype=torch.float32, device=self.device)

        texture_path = os.path.join(self.package_dirpath , 'data/data_3dmm/topology/Head.bmp')
        img = Image.open(texture_path).convert('RGB')  # Convert to RGB just in case it's not
        texture_np = np.array(img).astype(np.float32) / 255.0  # Shape: [H, W, 3]
        self.texture = torch.tensor(texture_np, dtype=torch.float32, device=self.device).unsqueeze(0)
        self.texture = self.texture.flip(1)

        # load lr mesh
        mesh = trimesh.load(os.path.join(self.multires_path,'level' + str(self.cfg['res_level']) + '.obj'),merge_norm=True, merge_tex=True)
        self.mean_face_vertices = mesh.vertices
        self.mean_face_triangles = mesh.faces

        # convert to torch
        self.mean_face_vertices = torch.tensor(mesh.vertices, dtype=torch.float32, device=self.device)
        self.mean_face_triangles = torch.tensor(mesh.faces, dtype=torch.int32, device=self.device)
        # self.mean_face_vertices,self.mean_face_triangles,self.duplicate_idx_lr = remove_duplicates(self.mean_face_vertices,self.mean_face_triangles)

        self.v_unique = self.mean_face_vertices.clone()
        self.f_unique = self.mean_face_triangles.clone()

        self.mesh = mesh
        return

    def init_multires(self):
        obj_to_decimate = self.cfg['face_path']
        self.multires_path = os.path.join(self.package_data_path , 'multires')

        low2high_dict_path = os.path.join(self.multires_path,'low2high.pt')
        if os.path.exists(low2high_dict_path) and not self.cfg['override_prev_decimate']: # if exists, so just load, otherwise create it
            # load low to high conversion
            info = torch.load(low2high_dict_path)
            transform_to_hr = info['low2high'][:self.cfg['res_level']]
            transform_to_hr = reduce(lambda a, b: a @ b, transform_to_hr)
            transform_to_hr = transform_to_hr.to(torch.float32).to('cuda')
            self.transform_to_hr = transform_to_hr
            return

        faces_c2f = []
        low2high = []
        for level in range(self.cfg['res_level']):
            if level != 0:
                obj_to_decimate = os.path.join(self.multires_path, 'level' + str(level) + '.obj')
            else:
                shutil.copy(obj_to_decimate , os.path.join(self.multires_path, 'level0' + '.obj'))
            low2high_curr = decimate_mesh_by_uv(obj_to_decimate, self.multires_path, high_level=level,
                                                      low_level=level+1, ratio=self.cfg['decimate_ratio'])
            low2high.append(low2high_curr.to_dense())
            # faces_c2f.append(faces_curr)

        low2high_dict = {}
        low2high_dict['low2high'] = low2high
        low2high_dict['faces_c2f'] = faces_c2f
        torch.save(low2high_dict, os.path.join(self.multires_path, 'low2high.pt'))
        transform_to_hr = low2high_dict['low2high'][:self.cfg['res_level']]
        transform_to_hr = reduce(lambda a, b: a @ b, transform_to_hr)
        transform_to_hr = transform_to_hr.to(torch.float32).to('cuda')
        self.transform_to_hr = transform_to_hr
        return

    def compute_symmetry_chamfer_loss(self, vertices, axis=0):
        """
        Computes Chamfer symmetry loss by comparing vertices to their mirrored version along a given axis.

        Parameters
        ----------
        vertices : torch.Tensor
            Tensor of shape (N, 3), the 3D vertex positions.
        axis : int
            Axis to reflect across (default: 0 = X axis)

        Returns
        -------
        sym_loss : torch.Tensor
            Scalar Chamfer loss enforcing symmetry along the specified axis.
        """
        # Clone and reflect
        v_reflected = vertices.clone()
        v_reflected[:, axis] *= -1

        # Compute Chamfer distance (bidirectional)
        dists = torch.cdist(vertices, v_reflected)
        loss_fwd = torch.min(dists, dim=1)[0].mean()
        loss_bwd = torch.min(dists, dim=0)[0].mean()
        sym_loss = loss_fwd + loss_bwd
        return sym_loss

    def optimize(self):
        v_unique_ori = self.v_unique.clone()
        f_unique = self.f_unique.clone()

        alpha = None

        # make contiguous TODO: is it necessary?
        v_unique = v_unique_ori.contiguous()
        f_unique = f_unique.contiguous()

        # update vertices of mean face
        self.mean_face_vertices = v_unique.clone()

        # create feature latent space for optimization
        text_ref_features_clip = self.create_ref_latent_vec()
        negative_prompt_vec = self.create_negative_prompt_vec()

        # add ref image
        vec_ref_imgs = self.add_ref_imgs()

        # Initialize the optimized variables and the optimizer
        translation = torch.zeros((1, 3), device=self.device, dtype=torch.float32)

        v_unique = v_unique_ori.clone()
        std_norm = self.cfg['mesh_scale_factor'] * torch.tensor(self.cfg['mesh_std_norm_scale']).to('cuda')
        self.mesh_offset = torch.tensor(self.cfg['mesh_offset']).to('cuda')
        offset = v_unique_ori.mean(0) + self.mesh_offset
        scale = std_norm/v_unique_ori.std(0)
        v_unique = (v_unique - offset) * scale
        v_unique_init = v_unique.clone()

        # initialize optimizer
        M = compute_matrix(v_unique, f_unique, lambda_=self.lambda_reg, alpha=alpha,cfg=self.cfg)
        u_unique = to_differential(M, v_unique)
        opt = initialize_optimizer(u_unique, v_unique, translation=None,use_translation=self.use_translation, step_size=self.step_size)
        if self.cfg['relative_loss']:
            img_first_render_latent = None # relative loss
        loss_clip_l1_arr = []

        # set threshold for maximum allowed deformation length wrt original std vertices
        deform_length_ratio_th = self.cfg.get('deform_length_t_ratio_h')
        deform_length_th = v_unique_init.std() * deform_length_ratio_th
        num_of_vertices = v_unique_init.numel()

        num_of_ite = self.cfg.get('num_of_iterations', 2101)
        for it in tqdm(range(num_of_ite), desc="Iteration Progress"):
            v_lr = from_differential(M, u_unique, self.solver)
            v_unique = v_lr

            v_deform = v_unique - v_unique_init
            v_deform_norm = torch.norm(v_deform, keepdim=True)
            v_deform_norm = v_deform_norm/num_of_vertices
            if v_deform_norm.squeeze() > deform_length_th:
                v_deform = deform_length_th * v_deform / v_deform_norm
                v_unique = v_unique_init + v_deform
                # print(v_deform_norm)

            # normalize mesh to stabilize convergence
            offset_curr = v_unique.mean(0) + self.mesh_offset
            scale_curr = std_norm / v_unique.std(0)
            v_unique = (v_unique - offset_curr) * scale_curr

            # Recompute vertex normals
            face_normals = compute_face_normals(v_unique, f_unique)
            n_opt = compute_vertex_normals(v_unique, f_unique, face_normals)

            if self.use_translation:
                opt_imgs = self.renderer.render(translation + v_unique, n_opt, f_unique)
            else:
                opt_imgs = self.renderer.render(v_unique, n_opt, f_unique)

            # add semantic loss
            img_opt_clip = self.normalize_for_clip(opt_imgs[:,:,:,:-1].permute((0,3,1,2)))
            img_opt_clip = transforms.functional.resize(img_opt_clip , (224,224) , antialias=True)
            img_opt_features_clip_abs = self.model_clip.encode_image(img_opt_clip)
            img_opt_features_clip =  img_opt_features_clip_abs / img_opt_features_clip_abs.norm(dim=-1, keepdim=True)

            # relative los
            if self.cfg['relative_loss'] and img_first_render_latent is None:
                img_first_render_latent = img_opt_features_clip_abs.detach().clone()
                img_first_render_latent /= img_first_render_latent.norm(dim=-1, keepdim=True)
                # vec_relative = text_ref_features_clip - img_first_render_latent

            # calculate semantic loss - cosine similarity
            im_loss = 0.0
            im_loss += (1 - torch.nn.CosineSimilarity(dim=1, eps=1e-08)(img_opt_features_clip,text_ref_features_clip)).mean()

            # calculate semantic loss - L1 norm
            loss_clip_l1 = 10* (img_opt_features_clip - text_ref_features_clip).abs()
            loss_clip_l1_arr.append(loss_clip_l1.detach().cpu().numpy())
            im_loss += loss_clip_l1.mean()

            if vec_ref_imgs is not None:
                im_loss_from_ref_imgs = (1 - torch.nn.CosineSimilarity(dim=1, eps=1e-08)(img_opt_features_clip,vec_ref_imgs)).mean()
                im_loss = self.alpha_text * im_loss + (1-self.alpha_text) * im_loss_from_ref_imgs

            # relative loss
            if self.cfg['relative_loss'] and img_first_render_latent is not None:
                # im_loss_relative = lambda_relvative_loss * torch.mean((vec_relative - (img_opt_features_clip_abs - img_first_render_latent))**2)
                im_loss_relative = self.cfg['lambda_relative_loss'] * (1 - torch.nn.CosineSimilarity(dim=1, eps=1e-08)(img_opt_features_clip,img_first_render_latent)).mean()
                im_loss += im_loss_relative.mean()

            loss_negative_prompt = torch.nn.CosineSimilarity(dim=1, eps=1e-08)(img_opt_features_clip,negative_prompt_vec).mean()
            im_loss += self.cfg['lambda_negative'] * loss_negative_prompt

            # calculate loss
            loss = im_loss

            # add symmetric loss
            if self.cfg.get('constraint_sym',True):
                sym_loss = self.compute_symmetry_chamfer_loss(v_unique, axis=0)
                loss += self.cfg.get('lambda_sym',10) * sym_loss

            if it % self.cfg['reinit_opt_every'] == 0:
                opt = initialize_optimizer(u_unique, v_unique, translation=None, use_translation=self.use_translation,step_size=self.step_size)
                # v_hr = self.transform_to_hr @ v_lr
                # vertices_hr_normalized = (self.vertices_hr - offset) * scale
                v_hr_deform = self.transform_to_hr @ (v_lr - v_unique_init)
                self.it = it
                self.save_mesh_hr(v_hr_deform,scale,offset,during_opt=True)
                self.save_imgs(self.loc_to_save_scene,opt_imgs,indices=[2,14])

            # Backpropagate
            opt.zero_grad()
            loss.backward()

            # Update parameters
            opt.step()

        # plt.imshow(opt_imgs[-1, :, :, :3].cpu().detach().numpy())
        # plt.show()

        # convert to original scale
        v_hr_deform = self.transform_to_hr @ (v_lr - v_unique_init)
        self.save_mesh_hr(v_hr_deform,scale,offset,during_opt=False)

        # save shapekeys in blender file
        create_blender_shapekeys = CreateBlenderShapekeys(input_folder=self.loc_to_save_scene,output_folder=self.loc_to_save_scene)
        create_blender_shapekeys()
        return

    def save_mesh_hr(self,v_hr_deform,scale,offset,during_opt=False):
        in_path = os.path.join(self.package_path,'data/multires/level0.obj')
        v_hr_new_deform = v_hr_deform/scale + offset

        v_hr_new = self.vertices_hr +  v_hr_new_deform
        v_hr_new = v_hr_new.detach().cpu().numpy()

        # check the existing folder to give count
        existing = []
        for d in os.listdir(self.loc_to_save_dir):
            full_dir = os.path.join(self.loc_to_save_dir, d)
            vertices_file = os.path.join(full_dir, "mesh_last.obj")
            if os.path.isdir(full_dir) and d.isdigit() and os.path.isfile(vertices_file):
                existing.append(d)
        existing_numbers = sorted(int(d) for d in existing) if existing else []

        # give next dir
        next_num = max(existing_numbers) + 1 if existing_numbers else 0
        self.loc_to_save_scene = os.path.join(self.loc_to_save_dir, f"{next_num:05d}")
        os.makedirs(self.loc_to_save_scene, exist_ok=True)
        # if os.path.exists(loc_to_save_scene):
        #     shutil.rmtree(vertices_file)
        print(f"Created: {self.loc_to_save_scene}")

        # save config
        save_path = os.path.join(self.loc_to_save_scene,"config.yaml")
        # add iteration to config
        cfg_to_save = self.cfg
        cfg_to_save["it"] = self.it
        with open(save_path, "w") as f:
            self.yaml.dump(cfg_to_save, f)
        print(f"Config saved to {save_path}")

        if during_opt:
            mesh_path = os.path.join(self.loc_to_save_scene,self.cfg['mesh_name'] + "_" + f"{self.it:04d}" + ".obj")
        else:
            mesh_path = os.path.join(self.loc_to_save_scene, "mesh_last.obj")
        replace_vertices(in_path, mesh_path, v_hr_new)
        self.loc_to_save_scene = self.loc_to_save_scene
        return

    def save_imgs(self,loc_to_save_scene,opt_imgs,indices=None):
        imgs_np = opt_imgs.cpu().detach().numpy()[:, :, :, :3]
        imgs_np = imgs_np[:, :, :, ::-1]
        if indices is None:
            indices = range(min(15, imgs_np.shape[0]))
        for i in indices:
            img_path = os.path.join(loc_to_save_scene, f"img_{i:02d}_{self.it:04d}.png")
            cv2.imwrite(img_path, (np.clip(255 * imgs_np[i],0,255)).astype('uint8'))
        return

    def __call__(self):
        self.optimize()
        return


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process scene filepath')
    parser.add_argument('--config_path', type=str, default='configs/config_multires_custom.yaml',
                        help='Name of the config file, relative to package folder')
    parser.add_argument('--scene_path', type=str, default='data/scenes/face/face.xml', help='Path to the scene file')
    parser.add_argument('--prompt', type=str, default=None, help='Path to the scene file')
    parser.add_argument('--step_size', type=float, default=None, help='Defines the step size used in the optimization process, controlling the magnitude of parameter updates per iteration')
    parser.add_argument('--num_of_iterations', type=int, default=None, help='number of iterations')
    parser.add_argument('--mesh_scale_factor', type=float, default=None, help='scale factor to render')
    parser.add_argument('--lambda_reg', type=float, default=None, help='Sobolev smoothness weight for deformation')
    parser.add_argument('--lambda_sym', type=float, default=None, help='Symmetric regularization weight for deformation')

    args = parser.parse_args()
    magic_shape_keys = MagicShapeKeysMultiRes(config_path=args.config_path,scene_path=args.scene_path,prompt=args.prompt,step_size=args.step_size,num_of_iterations=args.num_of_iterations,mesh_scale_factor=args.mesh_scale_factor,lambda_reg=args.lambda_reg,lambda_sym=args.lambda_sym)
    magic_shape_keys()

    print('DONE')