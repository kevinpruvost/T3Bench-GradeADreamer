import argparse
import os
import glob

def launch_command(command):
    print(command)
    os.system(command)

def train(args):
    fmt_prompt = args.prompt.replace(' ', '_')

    if args.stage == "1":
        launch_command(f'CUDA_VISIBLE_DEVICES={args.gpu} python Balanced_Score_Distillation/exp_2d.py --num_steps 1200 --log_steps 72 \
        --seed 1222 --lr 0.06 --phi_lr 0.0001 --use_t_phi true \
        --model_path stabilityai/stable-diffusion-2-1-base \
        --t_schedule random --generation_mode bsd --phi_model lora \
        --lora_scale 1. --lora_vprediction false \
        --prompt "{args.prompt}" \
        --height 512 --width 512 --batch_size 1 \
        --guidance_scale 7.5 --log_progress true \
        --save_x0 true --save_phi_model true --work_dir final/')

    elif args.stage == "2":
        launch_command(f'backgroundremover -i ./final/{fmt_prompt}/final_image_{fmt_prompt}.png -o ./final/{fmt_prompt}/final_image_{fmt_prompt}_no_alpha.png')
        launch_command(f'python imagetomesh.py --elev 10 --im_path ./final/{fmt_prompt}/final_image_{fmt_prompt}_no_alpha.png')

    elif args.stage == "3":
        launch_command(f'python train_dreambooth_lora.py \
        --pretrained_model_name_or_path=stabilityai/stable-diffusion-2-1-base  \
        --instance_data_dir=./multiview_images/final_image_{fmt_prompt}_no_alpha \
        --output_dir=./lora_checkpoints/{fmt_prompt}_no_alpha \
        --instance_prompt="{args.prompt}" \
        --class_prompt="A cactus" \
        --resolution=512 \
        --train_batch_size=2 \
        --gradient_accumulation_steps=1 \
        --learning_rate=1e-4 \
        --lr_scheduler="constant" \
        --lr_warmup_steps=0 \
        --max_train_steps=400 \
        --pre_compute_text_embeddings')

        launch_command(f'python train.py --opt configs/cactus.yaml \
            --lambda_ 18.0 \
            --name "{fmt_prompt}" \
            --lora_path lora_checkpoints/{fmt_prompt}_no_alpha'
        )

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--prompt', required=True)
    parser.add_argument('--gpu', required=True)
    parser.add_argument('--stage', required=True)
    args = parser.parse_args()

    train(args)
    