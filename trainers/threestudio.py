import argparse
import os
import glob

def first_pass(args, model_name):
    print(f'python launch.py --config configs/{model_name}.yaml --train --gpu {args.gpu} system.prompt_processor.prompt="{args.prompt}"')
    os.system(f'python launch.py --config configs/{model_name}.yaml --train --gpu {args.gpu} system.prompt_processor.prompt="{args.prompt}"')
    d = sorted(glob.glob(os.path.join(f'outputs/{model_name}', args.prompt.replace(' ', '_') + '*')))[-1].replace('\'', '\\\'')
    return d

def other_pass(args, model_name, d):
    print(f'python launch.py --config configs/{model_name}.yaml --train --gpu {args.gpu} system.prompt_processor.prompt={args.prompt} system.geometry_convert_from={d}/last.ckpt')
    os.system(f'python launch.py --config configs/{model_name}.yaml --train --gpu {args.gpu} system.prompt_processor.prompt={args.prompt} system.geometry_convert_from={d}/last.ckpt')
    d = sorted(glob.glob(os.path.join(f'outputs/{model_name}', args.prompt.replace(' ', '_') + '*')))[-1].replace('\'', '\\\'')
    return d

def export_mesh(path):
    print(f'python launch.py --config {path}/configs/parsed.yaml --export --gpu 0 resume={path}/ckpts/last.ckpt system.exporter_type=mesh-exporter system.exporter.fmt=obj-mtl')
    os.system(f'python launch.py --config {path}/configs/parsed.yaml --export --gpu 0 resume={path}/ckpts/last.ckpt system.exporter_type=mesh-exporter system.exporter.fmt=obj-mtl')

def prolificdreamer_train(args):
    # first pass
    d = first_pass(args, 'prolificdreamer')
    # other passes
    d = other_pass(args, 'prolificdreamer-geometry', d)
    d = other_pass(args, 'prolificdreamer-texture', d)
    # export mesh
    export_mesh(d)

def magic3d_train(args):
    # first pass
    d = first_pass(args, 'magic3d-coarse-sd')
    # other passes
    d = other_pass(args, 'magic3d-refine-sd', d)
    # export mesh
    export_mesh(d)

def fantasia3d_train(args):
    # first pass
    d = first_pass(args, 'fantasia3d')
    # other passes
    d = other_pass(args, 'fantasia3d-texture', d)
    # export mesh
    export_mesh(d)

def dreamfusion_train(args):
    # first pass
    d = first_pass(args, 'dreamfusion-sd')
    # export mesh
    export_mesh(d)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--prompt', required=True)
    parser.add_argument('--gpu', required=True)
    parser.add_argument('--model', required=True)
    args = parser.parse_args()

    # import module given in args to get extract_mesh
    if args.model == 'ProlificDreamer':
        prolificdreamer_train(args)
    elif args.model == 'Magic3D':
        magic3d_train(args)
    elif args.model == 'Fantasia3D':
        fantasia3d_train(args)
    elif args.model == 'DreamFusion':
        dreamfusion_train(args)
    else:
        raise ValueError(f'Unknown model: {args.model}')
    