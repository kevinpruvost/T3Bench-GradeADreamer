import os
import glob
import argparse

def extract_mesh_from_obj(mesh_obj_path, prompt, output_dir_name):
    """
    Extracts all files associated with the given .obj mesh file path.
    
    Args:
        mesh_obj_path (str): Path to the .obj file.
        prompt (str): Prompt used to generate the mesh.
        output_dir_name (str): Name of the directory to save the extracted files (in outputs_mesh_t3/).
    
    Returns:
        list: List of paths to the .obj file, its .mtl files, and associated textures.
    """
    prompt = prompt.replace(' ', '_')
    output_dir = f'outputs_mesh_t3/{output_dir_name}/{prompt}'
    associated_files = set()  # Use a set to avoid duplicates

    # Check if the provided path is a valid .obj file
    if not os.path.isfile(mesh_obj_path) or not mesh_obj_path.lower().endswith('.obj'):
        raise ValueError("Provided path is not a valid .obj file")

    # Add the .obj file itself
    associated_files.add(mesh_obj_path)
    
    # Extract the directory of the .obj file
    obj_dir = os.path.dirname(mesh_obj_path)

    # Find all .mtl files referenced in the .obj file
    mtl_files = set()
    with open(mesh_obj_path, 'r') as file:
        for line in file:
            if line.lower().startswith('mtllib'):
                mtl_filenames = line.split()[1:]
                for mtl_filename in mtl_filenames:
                    mtl_file = os.path.join(obj_dir, mtl_filename.strip())
                    if os.path.isfile(mtl_file):
                        mtl_files.add(mtl_file)

    # Add .mtl files to associated files
    associated_files.update(mtl_files)
    
    # Define common texture file extensions
    texture_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tga', '.tiff', '.gif'}

    # Find all texture files referenced in the .mtl files
    for mtl_file in mtl_files:
        with open(mtl_file, 'r') as file:
            for line in file:
                texture_filename = line.split()[1].strip()
                texture_file = os.path.join(obj_dir, texture_filename)
                
                # Check if the file has a valid texture extension
                if any(texture_file.lower().endswith(ext) for ext in texture_extensions) and os.path.isfile(texture_file):
                    associated_files.add(texture_file)
    
    files_list = list(associated_files)

    # Create the output directory if it does not exist
    os.makedirs(output_dir, exist_ok=True)
    for file_path in files_list:
        file_name = os.path.basename(file_path)
        #if .obj file then, rename to mesh.obj
        if file_path.endswith('.obj'):
            file_name = 'mesh.obj'
        output_path = os.path.join(output_dir, file_name)
        with open(file_path, 'rb') as src, open(output_path, 'wb') as dest:
            dest.write(src.read())
        print(f'Copied {file_path} to {output_dir}...')
