from PIL import Image
import argparse
import os
import numpy as np
import glob
from copy import deepcopy

from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch

# Load model and tokenizer
model_name = "Qwen/CodeQwen1.5-7B-Chat"  # or your chosen model

prompt_to_llm_ori = '''You are an assessment expert responsible for prompt-prediction pairs. Your task is to score the prediction according to the following requirements:
Evaluate the recall, or how well the prediction covers the information in the prompt.

1. If the prediction contains information that does not appear in the prompt, it should not be considered as bad.
2. If the prediction contains correct information about color or features in the prompt, you should also consider raising your score.
3. Assign a score between 1 and 5, with 5 being the highest. Do not provide a complete answer; give the score in the format: 5
'''

def predict(model_llm, pipe_llm, tokenizer, prompt, temperature):
    chat = [
        {"role": "system", "content": prompt_to_llm_ori},
        {"role": "user", "content": prompt}
    ]

    with torch.no_grad():
        outputs = pipe_llm(chat, max_new_tokens=512, do_sample=False, temperature=temperature)
    print(outputs[0]['generated_text'][2]['content'])
    return outputs[0]['generated_text'][2]['content']

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
    pipe_llm = pipeline("text-generation", model=model_llm, tokenizer=tokenizer)
    mean_score = 0
    output = ''

    with open(f'outputs_caption/{args.method}.txt') as f:
        lines = f.readlines()

    os.makedirs(f'outputs_alignment', exist_ok=True)
    
    for line in lines:
        line = line.strip()
        prompt, text = line.split(':', 1)
        print(prompt, text)
        text = process_text(text)

        prompt_to_llm = 'Prompt: ' + prompt + '\n'
        prompt_to_llm += 'Prediction: ' + text
        prompt_to_llm += '\nOnly give a number and nothing else. For example, if you think the prediction is a 3, type 3.'
        res = predict(model_llm, pipe_llm, tokenizer, prompt_to_llm, 0)
    
        print(res)
        mean_score += np.round(float(res)) / len(lines)
        output += f'{np.round(float(res)):.0f}\t\t{prompt}\n'

    print("Alignment Score:", mean_score)

    with open(f'outputs_alignment/{args.method}.txt', 'a+') as f:
        f.write(output)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--method', required=True, type=str, choices=['magic3d', 'fantasia3d', 'dreamfusion', 'prolificdreamer', 'gradeadreamer'])
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