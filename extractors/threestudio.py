from utils.mesh_extractor_utils import extract_mesh_from_obj
import os
import glob

def extract_mesh(prompt, formatted_prompt, model_name):
    if model_name == "prolificdreamer":
        model_name_ = "prolificdreamer-texture"
    elif model_name == "magic3d":
        model_name_ = "magic3d-refine-sd"
    elif model_name == "fantasia3d":
        model_name_ = "fantasia3d-texture"
    elif model_name == "dreamfusion":
        model_name_ = "dreamfusion-sd"
    else:
        raise ValueError(f"Model {model_name} not supported")
    d = sorted(glob.glob(os.path.join(f'./third_party/threestudio/outputs/{model_name_}', formatted_prompt + '*')))[-1].replace('\'', '\\\'')
    # find folder that contains it*-export in its name inside d
    mesh_folder = glob.glob(f"{d}/save/it*-export/")[0]
    mesh_path = f'{mesh_folder}model.obj'
    print(f"Found mesh at: {mesh_path}...")
    extract_mesh_from_obj(mesh_path, prompt, model_name)