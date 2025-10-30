import random

import torch
import numpy as np
import nvdiffrast.torch as dr
from flexigon.render_engine.render_utils import SphericalHarmonics, persp_proj

class NVDRenderer:
    """
    Renderer using nvdiffrast.


    This class encapsulates the nvdiffrast renderer [Laine et al 2020] to render
    objects given a number of viewpoints and rendering parameters.
    """
    def __init__(self, scene_params, shading=True, boost=1.0):
        """
        Initialize the renderer.

        Parameters
        ----------
        scene_params : dict
            The scene parameters. Contains the envmap and camera info.
        shading: bool
            Use shading in the renderings, otherwise render silhouettes. (default True)
        boost: float
            Factor by which to multiply shading-related gradients. (default 1.0)
        """
        # We assume all cameras have the same parameters (fov, clipping planes)
        near = scene_params["near_clip"]
        far = scene_params["far_clip"]
        self.fov_x = scene_params["fov"]
        w = scene_params["res_x"]
        h = scene_params["res_y"]
        self.res = (h,w)
        ar = w/h
        # x = torch.tensor([[1,2,3,4]], device='cuda')
        self.proj_mat = persp_proj(self.fov_x, ar, near, far)

        # Construct the Model-View-Projection matrices
        self.view_mats = torch.stack(scene_params["view_mats"])
        # self.mvps = self.proj_mat @ self.view_mats
        if 'cam_proj_mat' in scene_params.keys():
            self.mvps = torch.from_numpy(np.load(scene_params['cam_proj_mat'])).to(torch.float32).to('cuda')
        if scene_params.get('flix_x',True):
            flip_x = torch.tensor([
                [-1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ], dtype=torch.float32, device='cuda:0').unsqueeze(0)
            flipped_mvps = flip_x.expand(self.mvps.shape[0], -1, -1) @ self.mvps  # broadcasting
            all_mvps = torch.cat([self.mvps, flipped_mvps], dim=0)
            self.mvps = all_mvps
        self.view_mats = self.proj_mat.inverse() @ self.mvps

        self.boost = boost
        self.shading = shading

        # Initialize rasterizing context
        self.glctx = dr.RasterizeGLContext()
        # Load the environment map
        w,h,_ = scene_params['envmap'].shape
        envmap = scene_params['envmap_scale'] * scene_params['envmap']
        # Precompute lighting
        self.sh = SphericalHarmonics(envmap)
        # Render background for all viewpoints once
        self.render_backgrounds(envmap)

    def render_backgrounds(self, envmap):
        """
        Precompute the background of each input viewpoint with the envmap.

        Params
        ------
        envmap : torch.Tensor
            The environment map used in the scene.
        """
        h,w = self.res
        pos_int = torch.arange(w*h, dtype = torch.int32, device='cuda')
        pos = 0.5 - torch.stack((pos_int % w, pos_int // w), dim=1) / torch.tensor((w,h), device='cuda')
        a = np.deg2rad(self.fov_x)/2
        r = w/h
        f = torch.tensor((2*np.tan(a),  2*np.tan(a)/r), device='cuda', dtype=torch.float32)
        rays = torch.cat((pos*f, torch.ones((w*h,1), device='cuda'), torch.zeros((w*h,1), device='cuda')), dim=1)
        rays_norm = (rays.transpose(0,1) / torch.norm(rays, dim=1)).transpose(0,1)
        rays_view = torch.matmul(rays_norm, self.view_mats.inverse().transpose(1,2)).reshape((self.view_mats.shape[0],h,w,-1))
        theta = torch.acos(rays_view[..., 1])
        phi = torch.atan2(rays_view[..., 0], rays_view[..., 2])
        envmap_uvs = torch.stack([0.75-phi/(2*np.pi), theta / np.pi], dim=-1)
        self.bgs = dr.texture(envmap[None, ...], envmap_uvs, filter_mode='linear').flip(1)
        self.bgs = 0.4 * torch.ones_like(self.bgs)
        self.bgs[..., -1] = 0 # Set alpha to 0

    def render(self, v, n, f , uvs=None,texture=None,v_col=None):
        """
        Render the scene in a differentiable way.

        Parameters
        ----------
        v : torch.Tensor
            Vertex positions
        n : torch.Tensor
            Vertex normals
        f : torch.Tensor
            Model faces

        Returns
        -------
        result : torch.Tensor
            The array of renderings from all given viewpoints
        """
        v_hom = torch.nn.functional.pad(v, (0,1), 'constant', 1.0)
        v_ndc = torch.matmul(v_hom, self.mvps.transpose(1,2))
        rast = dr.rasterize(self.glctx, v_ndc, f, self.res)[0]
        if self.shading and texture is not None and uvs is not None:
            # === Textured shading with spherical harmonics ===

            # Interpolate UVs and normals
            uv_interp = dr.interpolate(uvs[None, ...], rast, f)[0]  # [B, H, W, 2]
            n_interp = dr.interpolate(n[None, ...], rast, f)[0]  # [B, H, W, 3]
            n_interp = torch.nn.functional.normalize(n_interp, dim=-1)

            # Sample lighting from SH per pixel
            sh_light = self.sh.eval(n_interp)  # [B, H, W, 3]

            # uv_interp and n_interp already computed above
            # Sample texture once (clamp to avoid wrap bleeding)
            tex_col = dr.texture(texture, uv_interp, filter_mode='linear', boundary_mode='clamp')  # [B, H, W, 3]

            # Lighting (already computed)
            lit_col = tex_col * sh_light  # [B, H, W, 3]

            # coverage / per-pixel alpha from rasterizer (values in [0,1])
            coverage = rast[..., -1:].clamp(0.0, 1.0)  # [B, H, W, 1]

            # premultiply RGB by coverage (important for correct blending)
            premul_rgb = lit_col * coverage  # [B, H, W, 3]
            premul_col = torch.cat((premul_rgb, coverage), dim=-1)  # [B, H, W, 4]

            # Ensure background is opaque (alpha=1) and has same shape
            # if you want solid bg color use e.g. torch.full(..., fill_value=1.0) in alpha
            bg = self.bgs.clone()
            if bg.shape[-1] == 4:
                bg_alpha = bg[..., -1:]
                bg_rgb = bg[..., :3]
                # make background opaque (avoid alpha=0 which can cause compositing fringe)
                bg = torch.cat((bg_rgb, torch.ones_like(bg_alpha)), dim=-1)
            else:
                # assume bgs is RGB -> make alpha channel = 1
                bg = torch.cat((bg, torch.ones((*bg.shape[:-1], 1), device=bg.device)), dim=-1)

            # Use torch.where to choose premultiplied object color where coverage>0, else background
            # (this ensures areas not covered are background; edges will have premultiplied partial alpha)
            img_in = torch.where(coverage > 0.0, premul_col, bg)

            # Call antialias with premultiplied RGBA input
            result = dr.antialias(img_in, rast, v_ndc, f, pos_gradient_boost=self.boost)

        elif self.shading:
            # === SH shading without texture (e.g. diffuse vertex lighting) ===

            if v_col is not None:
                vert_light = v_col * self.sh.eval(n).contiguous()
            else:
                vert_light = self.sh.eval(n).contiguous()

            # Interpolate lighting to pixels
            light = dr.interpolate(vert_light[None, ...], rast, f)[0]

            # Add alpha channel
            col = torch.cat((light / np.pi, torch.ones((*light.shape[:-1], 1), device='cuda')), dim=-1)

            # Apply anti-aliasing and background
            result = dr.antialias(torch.where(rast[..., -1:] != 0, col, self.bgs), rast, v_ndc, f,
                                  pos_gradient_boost=self.boost)

        else:
            # === Flat rendering with vertex colors only ===

            if v_col is None:
                v_col = torch.ones_like(v)

            # Interpolate vertex colors to pixels
            col = dr.interpolate(v_col[None, ...], rast, f)[0]

            # Apply anti-aliasing (no background used here)
            result = dr.antialias(col, rast, v_ndc, f, pos_gradient_boost=self.boost)

        return torch.flip(result , [1])

    def render_depth(self , v , f):
        num_layers = 1
        v_hom = torch.nn.functional.pad(v, (0,1), 'constant', 1.0)
        v_ndc = torch.matmul(v_hom, self.mvps.transpose(1,2))
        with dr.DepthPeeler(self.glctx, v_ndc, f, self.res) as peeler:
            for i in range(num_layers):
                rast, rast_db = peeler.rasterize_next_layer()
        return rast