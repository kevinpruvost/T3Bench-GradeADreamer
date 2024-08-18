import argparse    

import torch

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--prompt', required=True)
    parser.add_argument('--module', required=True)
    parser.add_argument('--model', required=False, default=None)
    parser.add_argument('--gpu', required=False, default="0")
    args = parser.parse_args()

    # import module given in args to get extract_mesh
    module = __import__(args.module)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(device)
    module.extract_mesh(args.prompt, args.prompt.replace(' ', '_'), args.model if args.model else args.module)