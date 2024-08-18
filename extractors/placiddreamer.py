from utils.mesh_extractor_utils import extract_mesh_from_obj
import os, sys
import glob

# Get the path to the 'third_party' directory
third_party_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'third_party/GradeADreamer'))

print(third_party_path)

# Add 'third_party' directory to sys.path
sys.path.insert(0, third_party_path)

from gradeadreamer.renderer.gs_renderer import GaussianModel
from gradeadreamer.utils.cam_utils import orbit_camera, OrbitCamera
from gradeadreamer.renderer.gs_renderer import MiniCam
from gradeadreamer.utils.mesh import Mesh, safe_normalize
from gradeadreamer.utils.grid_put import mipmap_linear_grid_put_2d
from gradeadreamer.renderer.gs_renderer import Renderer, MiniCam
import nvdiffrast.torch as dr
import torch
import numpy as np
import torch.nn.functional as F


def extract_texture(renderer, mesh, cam, path):
    texture_size = 1024
    # perform texture extraction
    print(f"[INFO] unwrap uv...")
    h = w = texture_size
    mesh.auto_uv()
    mesh.auto_normal()

    albedo = torch.zeros((h, w, 3), device="cuda", dtype=torch.float32)
    cnt = torch.zeros((h, w, 1), device="cuda", dtype=torch.float32)

    vers = [0] * 8 + [-45] * 8 + [45] * 8 + [-89.9, 89.9]
    hors = [0, 45, -45, 90, -90, 135, -135, 180] * 3 + [0, 0]

    render_resolution = 512

    glctx = dr.RasterizeCudaContext()

    for ver, hor in zip(vers, hors):
        # render image
        pose = orbit_camera(ver, hor, cam.radius)

        cur_cam = MiniCam(
            pose,
            render_resolution,
            render_resolution,
            cam.fovy,
            cam.fovx,
            cam.near,
            cam.far,
        )
        
        cur_out = renderer.render(cur_cam)

        rgbs = cur_out["image"].unsqueeze(0) # [1, 3, H, W] in [0, 1]

        # enhance texture quality with zero123 [not working well]
        # if self.opt.guidance_model == 'zero123':
        #     rgbs = self.guidance.refine(rgbs, [ver], [hor], [0])
            # import kiui
            # kiui.vis.plot_image(rgbs)
            
        # get coordinate in texture image
        pose = torch.from_numpy(pose.astype(np.float32)).to("cuda")
        proj = torch.from_numpy(cam.perspective.astype(np.float32)).to("cuda")

        v_cam = torch.matmul(F.pad(mesh.v, pad=(0, 1), mode='constant', value=1.0), torch.inverse(pose).T).float().unsqueeze(0)
        v_clip = v_cam @ proj.T
        rast, rast_db = dr.rasterize(glctx, v_clip, mesh.f, (render_resolution, render_resolution))

        depth, _ = dr.interpolate(-v_cam[..., [2]], rast, mesh.f) # [1, H, W, 1]
        depth = depth.squeeze(0) # [H, W, 1]

        alpha = (rast[0, ..., 3:] > 0).float()

        uvs, _ = dr.interpolate(mesh.vt.unsqueeze(0), rast, mesh.ft)  # [1, 512, 512, 2] in [0, 1]

        # use normal to produce a back-project mask
        normal, _ = dr.interpolate(mesh.vn.unsqueeze(0).contiguous(), rast, mesh.fn)
        normal = safe_normalize(normal[0])

        # rotated normal (where [0, 0, 1] always faces camera)
        rot_normal = normal @ pose[:3, :3]
        viewcos = rot_normal[..., [2]]

        mask = (alpha > 0) & (viewcos > 0.5)  # [H, W, 1]
        mask = mask.view(-1)

        uvs = uvs.view(-1, 2).clamp(0, 1)[mask]
        rgbs = rgbs.view(3, -1).permute(1, 0)[mask].contiguous()
        
        # update texture image
        cur_albedo, cur_cnt = mipmap_linear_grid_put_2d(
            h, w,
            uvs[..., [1, 0]] * 2 - 1,
            rgbs,
            min_resolution=256,
            return_count=True,
        )
        
        # albedo += cur_albedo
        # cnt += cur_cnt
        mask = cnt.squeeze(-1) < 0.1
        albedo[mask] += cur_albedo[mask]
        cnt[mask] += cur_cnt[mask]

    mask = cnt.squeeze(-1) > 0
    albedo[mask] = albedo[mask] / cnt[mask].repeat(1, 3)

    mask = mask.view(h, w)

    albedo = albedo.detach().cpu().numpy()
    mask = mask.detach().cpu().numpy()

    # dilate texture
    from sklearn.neighbors import NearestNeighbors
    from scipy.ndimage import binary_dilation, binary_erosion

    inpaint_region = binary_dilation(mask, iterations=32)
    inpaint_region[mask] = 0

    search_region = mask.copy()
    not_search_region = binary_erosion(search_region, iterations=3)
    search_region[not_search_region] = 0

    search_coords = np.stack(np.nonzero(search_region), axis=-1)
    inpaint_coords = np.stack(np.nonzero(inpaint_region), axis=-1)

    knn = NearestNeighbors(n_neighbors=1, algorithm="kd_tree").fit(
        search_coords
    )
    _, indices = knn.kneighbors(inpaint_coords)

    albedo[tuple(inpaint_coords.T)] = albedo[tuple(search_coords[indices[:, 0]].T)]

    mesh.albedo = torch.from_numpy(albedo).to("cuda")
    mesh.write(path)
    return mesh

def extract_mesh(prompt, formatted_prompt, model_name):
    renderer = Renderer(sh_degree=0)
    ply_path = f'./third_party/PlacidDreamer/output/{formatted_prompt}/point_cloud/iteration_5000/point_cloud.ply'
    output_path = f'./third_party/PlacidDreamer/output/{formatted_prompt}/point_cloud/iteration_5000/point_cloud.obj'
    print("hello")
    renderer.gaussians.load_ply(ply_path)
    print("hello1")
    mesh = renderer.gaussians.extract_mesh(output_path)
    print("hello2")
    cam = OrbitCamera(800, 800, 2.5, 49.1)
    extract_texture(renderer, mesh, cam, output_path)

    print(f"Found mesh at: {output_path}...")
    extract_mesh_from_obj(output_path, prompt, model_name)