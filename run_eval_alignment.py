from PIL import Image
import argparse
import os
import numpy as np
import glob
from copy import deepcopy

from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch

# Load model and tokenizer
model_name = "NousResearch/Meta-Llama-3-8B-Instruct"  # or your chosen model

def predict(model_llm, tokenizer, prompt, temperature):
    inputs = tokenizer(prompt, return_tensors="pt")

    with torch.no_grad():
        outputs = model_llm.generate(
            **inputs,
            max_new_tokens=15,  # adjust as needed
            temperature=temperature,
            do_sample=True,
            top_p=0.6,
        )
    
    full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Extract only the model's response
    answer = full_response[len(prompt):].strip()
    print(f"full_response: {answer}")
    return answer

def process_text(text):
    if text[-1] == '.':
        text = text[:-1]
    if text[0] == '"' and text[-1] == '"':
        text = text[1:-1]
    if text[0] == '\'' and text[-1] == '\'':
        text = text[1:-1]
    if text[-1] == '.':
        text = text[:-1]
    return text

def run(args, temp_path):
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model_llm = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")

    mean_score = 0
    output = ''

    prompt_to_llm_ori = '''You are an assessment expert responsible for prompt-prediction pairs. Your task is to score the prediction according to the following requirements:

    1. Evaluate the recall, or how well the prediction covers the information in the prompt. If the prediction contains information that does not appear in the prompt, it should not be considered as bad.
    2. If the prediction contains correct information about color or features in the prompt, you should also consider raising your score.
    3. Assign a score between 1 and 5, with 5 being the highest. Do not provide a complete answer; give the score in the format: 3

    '''

    with open(f'outputs_caption/{args.method}_{args.group}.txt') as f:
        lines = f.readlines()

    os.makedirs(f'result/alignment', exist_ok=True)
    
    for line in lines:
        line = line.strip()
        prompt, text = line.split(':', 1)
        print(prompt, text)
        text = process_text(text)

        prompt_to_llm = prompt_to_llm_ori
        prompt_to_llm += 'Prompt: ' + prompt + '\n'
        prompt_to_llm += 'Prediction: ' + text
        prompt_to_llm += '\nOnly give a number and nothing else. For example, if you think the prediction is a 3, type 3.'
        print(prompt_to_llm)
        res = predict(model_llm, tokenizer, prompt_to_llm, 0.6)
    
        print(res)
        mean_score += np.round(float(res)) / len(lines)
        output += f'{np.round(float(res)):.0f}\t\t{prompt}\n'

    print("Alignment Score:", mean_score)

    with open(f'result/alignment/{args.method}_{args.group}.txt', 'a+') as f:
        f.write(output)

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