from utils.mesh_extractor_utils import extract_mesh_from_obj

def extract_mesh(prompt, formatted_prompt, model_name):
    mesh_path = f'./third_party/GradeADreamer/logs/{formatted_prompt}/mesh.obj'
    extract_mesh_from_obj(mesh_path, prompt, model_name)