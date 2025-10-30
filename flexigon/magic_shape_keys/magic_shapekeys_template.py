import argparse
import os
import torch
import flexigon
from flexigon.render_engine.render import NVDRenderer
from flexigon.render_engine.load_xml import load_scene
from flexigon.geometry_processing.geometry_utils import remove_duplicates, compute_face_normals, compute_vertex_normals, average_edge_length
from flexigon.optimizers.optimizer_utils import initialize_optimizer
from flexigon.optimizers.Adam import AdamUniform
from flexigon.geometry_processing.sobolev_utils import compute_matrix, laplacian_uniform,laplacian_cot
from flexigon.geometry_processing.parameterize import to_differential, from_differential
import matplotlib.pyplot as plt

import clip
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR100
import cv2
from ruamel.yaml import YAML
from ruamel.yaml.comments import CommentedMap, CommentedSeq

class MagicShapeKeys:
    def __init__(self , config_path,scene_path,**kwargs):
        # device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Get the package path
        self.package_path = flexigon.__path__[0]
        self.package_dirpath = os.path.dirname(self.package_path)
        self.package_data_path = os.path.join(self.package_dirpath, 'data')

        # set absolute path for scene_path
        scene_full_path = os.path.join(self.package_dirpath, scene_path)

        # Convert config relative path to absolute path
        script_dir = os.path.dirname(os.path.abspath(__file__))
        if not os.path.isabs(config_path):
            config_path = os.path.join(script_dir, config_path)

        # Initialize YAML handler
        self.yaml = YAML()
        self.yaml.preserve_quotes = True  # keep original quotes if they exist

        # Load the config
        with open(config_path, 'r') as f:
            config = self.yaml.load(f)  # use .load(), not .safe_load()
        config = self.override_config(config,**kwargs)

        # change relative path to absolute path
        config = self.transform_relative_path_to_abs_path(config)

        # set saved results folder
        self.loc_to_save_dir = os.path.join(self.package_dirpath , 'results', config.get('output_folder_name','saved_results'))
        os.makedirs(self.loc_to_save_dir, exist_ok=True)

        # set default values if not provided
        self.init_default_values(config)

        # store the passed configuration dictionary as an instance attribute
        self.cfg = config
        self.check_cfg_availability()

        # Assign each config parameter as an instance attribute
        for key, value in config.items():
            setattr(self, key, value)

        #
        self.lambda_reg = float(self.lambda_reg)

        # Load the scene
        self.load_scene(scene_full_path)
        self.load_rendiff_engine()
        return

    def override_config(self, config, **kwargs):
        """
        Update the given config dictionary with any keyword arguments passed.
        """
        for key, value in kwargs.items():
            if value is not None:  # only override if value is not None
                config[key] = value
        return config

    def transform_relative_path_to_abs_path(self,cfg):
        """Recursively transforms relative calibration paths to absolute paths in-place, preserving quotes."""
        project_root = os.path.dirname(flexigon.__path__[0])
        replacements = {
            '<project_root>': project_root,
        }

        def _replace_in_obj(obj):
            if isinstance(obj, str):
                for placeholder, real_value in replacements.items():
                    obj = obj.replace(placeholder, real_value)
                return obj
            elif isinstance(obj, CommentedMap):
                for k, v in obj.items():
                    obj[k] = _replace_in_obj(v)
                return obj
            elif isinstance(obj, CommentedSeq) or isinstance(obj, list):
                for i, v in enumerate(obj):
                    obj[i] = _replace_in_obj(v)
                return obj
            else:
                return obj

        _replace_in_obj(cfg)
        return cfg

    def init_default_values(self,config):
        # add default values if not provided
        config.setdefault("boost", 1) # Gradient boost used in nvdiffrast
        config.setdefault("shading", 1) # Use shading in render engine, otherwise render silhouettes
        config.setdefault("laplacian_type", "robust") # laplacian type - uniform, cot or robust
        config.setdefault("optimize_translation", True)
        config.setdefault("use_translation", False)
        config.setdefault("step_size", 0.005)
        config.setdefault("solver", 'Cholesky') # Solver to use
        config.setdefault("prompt", None) # Solver to use
        config.setdefault("negative_prompt", None)
        config.setdefault("lambda_reg", 5e-3) # Solver to use
        config.setdefault("ref_imgs_path", None) # Solver to use
        config.setdefault("alpha_text", 0.5)  # Solver to use
        if config["ref_imgs_path"] is None:
            config['alpha_text'] = 0

        # optimizer
        self.optimizer = config.get("optimizer", AdamUniform)  # Which optimizer to use
        return

    def check_cfg_availability(self):
        assert self.cfg['laplacian_type'] in {'uniform', 'cot', 'robust'}, 'laplacian type must be uniform, cot or robust'
        return

    def load_scene(self,scene_full_path):
        # Load the scene
        self.scene_params = load_scene(scene_full_path)

        # make paths to absolute path
        self.scene_params['cam_proj_mat'] = os.path.join(self.package_dirpath, 'data',self.scene_params['cam_proj_mat'])
        return

    def load_rendiff_engine(self):
        # Initialize the renderer
        self.renderer = NVDRenderer(self.scene_params, shading=self.cfg['shading'], boost=self.cfg['boost'])
        return

    def create_ref_latent_vec(self):
        self.cifar100 = CIFAR100(root=os.path.expanduser("~/.cache"), download=True, train=False)
        self.model_clip, self.preprocess = clip.load("ViT-B/32", device=self.device)
        self.normalize_for_clip = transforms.Normalize([0.48145466, 0.4578275, 0.40821073], [0.26862954, 0.26130258, 0.27577711])

        if self.prompt is None:
            self.cifar100.classes.append('symmetrical photorealistic face')
        else:
            self.cifar100.classes.append(self.prompt)

        self.text_inputs = torch.cat([clip.tokenize(f"a photo of {c}") for c in self.cifar100.classes]).to(self.device)

        # Calculate features
        with torch.no_grad():
            self.text_ref_features_clip = self.model_clip.encode_text(self.text_inputs)
            self.text_ref_features_clip = (self.text_ref_features_clip / self.text_ref_features_clip.norm(dim=-1, keepdim=True))
            c = 100  # 21 #100 #21
            # [(idx,c) for idx,c in enumerate(cifar100.classes) if '' in c]
            print(f"Reference class index: {c}, class name: '{self.cifar100.classes[c]}'")
            img_ref_features_clip = self.text_ref_features_clip[c:c + 1, :]
            # similarity_ref = (100 * img_ref_features_clip @ text_ref_features_clip.T).softmax(dim=-1)
        return img_ref_features_clip

    def add_ref_imgs(self):
        if self.cfg['ref_imgs_path'] is None:
            return None

        # load reference image
        ref_img = cv2.imread(self.cfg['ref_imgs_path'])
        ref_img = cv2.cvtColor(ref_img, cv2.COLOR_BGR2RGB)
        ref_img = torch.from_numpy(ref_img).permute((2, 0, 1)).unsqueeze(0).float() / 255.0
        ref_img = ref_img.to(self.device)

        # Calculate features
        with torch.no_grad():
            # add semantic loss
            img_opt_clip = self.normalize_for_clip(ref_img)
            img_opt_clip = transforms.functional.resize(img_opt_clip , (224,224) , antialias=True)
            img_opt_features_clip = self.model_clip.encode_image(img_opt_clip)
            img_opt_features_clip =  img_opt_features_clip / img_opt_features_clip.norm(dim=-1, keepdim=True)

        return img_opt_features_clip

    def create_negative_prompt_vec(self):
        if self.negative_prompt is None:
            return None
        self.text_inputs = torch.cat([clip.tokenize(self.negative_prompt)]).to(self.device)

        # Calculate features
        with torch.no_grad():
            self.text_neg_features_clip = self.model_clip.encode_text(self.text_inputs)

        return self.text_neg_features_clip

    def optimize(self):
        v_unique_ori = self.scene_params['mesh-source']['vertices']
        f_unique = self.scene_params['mesh-source']['faces']
        alpha = None

        std_norm = torch.tensor((0.05, 0.08, 0.06)).to('cuda') / 1.1
        v_unique = (v_unique_ori - v_unique_ori.mean(0)) / v_unique_ori.std(0) * std_norm

        # make contiguous TODO: is it necessary?
        v_unique = v_unique.contiguous()
        f_unique = f_unique.contiguous()

        # create feature latent space for optimization
        img_ref_features_clip = self.create_ref_latent_vec()
        negative_prompt_vec = self.create_negative_prompt_vec()

        # Initialize the optimized variables and the optimizer
        translation = torch.zeros((1, 3), device=self.device, dtype=torch.float32)

        # initialize optimizer
        M = compute_matrix(v_unique, f_unique, lambda_=self.lambda_reg, alpha=alpha)
        u_unique = to_differential(M, v_unique)
        opt = initialize_optimizer(u_unique, v_unique, translation=None,use_translation=self.use_translation, step_size=self.step_size)
        for it in range(2000):
            # Get cartesian coordinates
            v_unique = from_differential(M, u_unique, self.solver)

            # define the optimized vertices
            if not self.use_translation:
                v_unique = (v_unique - v_unique.mean(0)) / v_unique.std(0) * std_norm
            else:
                v_unique = v_unique + translation

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
            img_opt_features_clip = self.model_clip.encode_image(img_opt_clip)
            img_opt_features_clip =  img_opt_features_clip / img_opt_features_clip.norm(dim=-1, keepdim=True)

            with torch.no_grad():
                # Pick the top 5 most similar labels for the image
                img_opt_features_clip_display = img_opt_features_clip.clone().detach()
                img_opt_features_clip_display /= img_opt_features_clip_display.norm(dim=-1, keepdim=True)
                # text_ref_features_clip /= text_ref_features_clip.norm(dim=-1, keepdim=True)
                similarity = (100.0 * img_opt_features_clip @ self.text_ref_features_clip.T).softmax(dim=-1)

                values, indices = similarity[0].topk(5)
                # Print the result
                print("\nTop predictions:\n")
                for value, index in zip(values, indices):
                    print(f"{self.cifar100.classes[index]:>16s}: {100 * value.item():.2f}%")

            im_loss = 0.0
            im_loss += (1 - torch.nn.CosineSimilarity(dim=1, eps=1e-08)(img_opt_features_clip,img_ref_features_clip)).mean()
            loss_negative_prompt = torch.nn.CosineSimilarity(dim=1, eps=1e-08)(img_opt_features_clip,negative_prompt_vec).mean()
            print(loss_negative_prompt.item())
            im_loss += loss_negative_prompt

            loss = im_loss

            # Backpropagate
            opt.zero_grad()
            loss.backward()

            # vertices_grad = u_unique.grad.cpu().detach().numpy()
            # np.save('/tmp/vertices_grad.npy', vertices_grad)

            # Update parameters
            opt.step()

        plt.imshow(opt_imgs[-1, :, :, :3].cpu().detach().numpy())
        plt.show()
        return

    def __call__(self):
        self.optimize()
        return

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process scene filepath')
    parser.add_argument('--config_name', type=str, default='config_heat',
                        help='Name of the config file, relative to package folder')
    parser.add_argument('--scene_path', type=str, default='data/scenes/face/face.xml', help='Path to the scene file')

    args = parser.parse_args()
    magic_shape_keys = MagicShapeKeys(config_name=args.config_name,scene_path=args.scene_path)
    magic_shape_keys()

    print('DONE')