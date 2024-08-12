from PIL import Image
import argparse
import os
import numpy as np
import glob
from copy import deepcopy

# import openai
#import backoff
from tqdm import tqdm
import trimesh

from lavis.models import load_model_and_preprocess


#openai.api_key = "PLEASE USE YOUR OWN API KEY"
#MODEL = "gpt-4"

#@backoff.on_exception(backoff.expo, (openai.error.RateLimitError, openai.error.Timeout, openai.error.APIError))
# def predict(prompt, temperature):
#     # message = openai.ChatCompletion.create(
#     #     model = MODEL,
#     #     temperature = temperature,
#     #     messages = [
#     #             # {"role": "system", "content": prompt}
#     #             {"role": "user", "content": prompt}
#     #         ]
#     # )
#     # # print(message)
#     # return message["choices"][0]["message"]["content"]
#     return

from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch

# Load model and tokenizer
model_name = "NousResearch/Meta-Llama-3-8B-Instruct"  # or your chosen model

prompt_to_llm_ori = 'Given a set of descriptions about the same 3D object, distill these descriptions into one concise caption. The descriptions are as follows:\n\n'
prompt_to_llm_ori += '\nAvoid describing background, surface, and posture and Answer directly with the distillation alone, no useless sentences.'

def predict(model_llm, tokenizer, pipe_llm, prompt, temperature):
    chat = [
        {"role": "system", "content": prompt_to_llm_ori},
        {"role": "user", "content": prompt}
    ]

    with torch.no_grad():
        outputs = pipe_llm(chat, max_new_tokens=512, do_sample=False, temperature=temperature)
    print(outputs[0]['generated_text'][2]['content'])
    return outputs[0]['generated_text'][2]['content']

    # inputs = tokenizer(prompt, return_tensors="pt")

    # with torch.no_grad():
    #     outputs = model_llm.generate(
    #         **inputs,
    #         max_new_tokens=15,  # adjust as needed
    #         temperature=temperature,
    #         do_sample=False,
    #         top_p=0.3,
    #     )
    
    # full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # # Extract only the model's response
    # answer = full_response[len(prompt):].strip()
    # return answer

# The rest of your code remains the same


def run(args, temp_path):
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model_llm = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")
    pipe_llm = pipeline("text-generation", model=model_llm, tokenizer=tokenizer)
    model, vis_processors, _ = load_model_and_preprocess(name='blip2_t5', model_type='pretrain_flant5xxl', is_eval=True, device=device)

    Radius = 2.2
    icosphere = trimesh.creation.icosphere(subdivisions=0)
    icosphere.vertices *= Radius

    with open(f'data/prompt_{args.group}.txt') as f:
        lines = f.readlines()

    os.makedirs(f'outputs_caption', exist_ok=True)
    
    # remove output file first
    try:
        os.remove(f'outputs_caption/{args.method}_{args.group}.txt')
    except FileNotFoundError:
        pass

    for prompt in lines:
        prompt = prompt.strip()
        try:
            obj_path = glob.glob(f'outputs_mesh_t3/{args.method}/{args.group}/mesh.obj')[-1].replace("\'", "\\\'")
        except IndexError:
            obj_path = 'outputs_mesh_t3/FALSE_PATH'
        
        os.system(f'python render/meshrender_cap.py --path {obj_path} --name {temp_path}')

        texts = []

        for idx, img_path in enumerate(os.listdir(temp_path)):
            color = Image.open(os.path.join(temp_path, img_path)).convert("RGB")
            image = vis_processors["eval"](color).unsqueeze(0).to(device)
            x = model.generate({"image": image}, use_nucleus_sampling=True, num_captions=1)
            texts += x
    
        prompt_input = ""
        for idx, txt in enumerate(texts):
            prompt_input += f'view{idx+1}: '
            prompt_input += txt
            prompt_input += '\n'
        res = predict(model_llm, tokenizer, pipe_llm, prompt_input, 0.6)

        with open(f'outputs_caption/{args.method}_{args.group}.txt', 'a+') as f:
            f.write(prompt + ':' + res + '\n')
        
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--group', type=str, default='single', choices=['astro'])
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--method', type=str, choices=['magic3d', 'fantasia3d', 'prolificdreamer', 'gradeadreamer'])
    args = parser.parse_args()

    i = 0
    while True:
        try: 
            temp_path = f'temp/temp_{i}'
            os.makedirs(temp_path)
            break
        except:
            i += 1
    
    run(args, temp_path)
    os.system(f'rm -r {temp_path}')
