import argparse    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--prompt', required=True)
    parser.add_argument('--module', required=True)
    parser.add_argument('--model', required=True)
    args = parser.parse_args()

    # import module given in args to get extract_mesh
    module = __import__(args.module)
    module.extract_mesh(args.prompt, args.prompt.replace(' ', '_'), args.model)