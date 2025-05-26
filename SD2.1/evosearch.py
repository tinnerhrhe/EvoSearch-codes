import torch
import os
from PIL import Image
from scheduling_ddim import DDIMScheduler
from diffusers import StableDiffusionPipeline
import pandas as pd
import numpy as np
import argparse
from datetime import datetime
from utils import do_eval
import json
from rewards import (
    do_clip_score,
    do_clip_score_diversity,
    do_image_reward,
    do_human_preference_score,
)
import math
from eval import load_images
from evo_pipe import EvolStableDiffusion
# torch.set_default_device('cpu')
def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--cfg', type=float, default=7.5, help='guidance for denoising process')
    parser.add_argument('--infer_step', type=int, default=50, help='total inference timestep T')
    parser.add_argument('--image_size', type=int, default=512, help='The size (height and width) of the generated image.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed to determine the initial latent.')
    parser.add_argument('--device', type=str, default='cuda', help='Device where the model inference is performed.')
    parser.add_argument('--model_name', type=str, default='stabilityai/stable-diffusion-2-1', help='pre-trained model name')
    parser.add_argument('--prompt_path', type=str, default='./DrawBench_Prompts.csv', help='prompt file path')
    parser.add_argument('--save_dir', type=str, default='./test', help='Path to save the generated images.')
    ### EvoSearch
    parser.add_argument('--evolution_schedule', nargs='+', type=int,help='Path to save the generated images.')
    parser.add_argument('--population_size_schedule', nargs='+', type=int, help='Path to save the generated images.')
    parser.add_argument('--guidance_reward', type=str,help='Guidance reward function used for search')
    parser.add_argument('--iterations', type=int, default=1, help='Iterations for doing evosearch over a batch of data')
    parser.add_argument('--number_of_N', type=int, default=100, help='scaling size')
    parser.add_argument('--mutation_rate', type=float, default=0.2, help='Mutation rate')
    parser.add_argument('--elite_size', type=int, default=5, help='Number of elites to keep across each generation')
    args = parser.parse_args()
    return args



def set_seed(random_seed):
    np.random.seed(int(random_seed))
    torch.manual_seed(int(random_seed))
    torch.cuda.manual_seed(int(random_seed))
    generator = torch.manual_seed(random_seed)
    return generator
    
def clean_filename(text):
    """Clean prompt text to create safe filenames"""
    return "".join(c if c.isalnum() else "_" for c in text)[:150]
# def evol_search(args):
def dynamic_population_size(n_min,n_max,nfe,nfe_max):
    size = math.ceil(n_max - (n_max-n_min)*(nfe/nfe_max))
    if size > (nfe_max-nfe):
        return nfe_max-nfe
    return size
if __name__ == '__main__':
    #load args
    args = get_args()
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir, exist_ok=True)
    #load model
    dtype = torch.float16
    device = torch.device(args.device)
    pipe = EvolStableDiffusion.from_pretrained(
        args.model_name,
        torch_dtype=dtype).to(device)
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to(args.device)
    if hasattr(pipe, "enable_xformers_memory_efficient_attention"):
        pipe.enable_xformers_memory_efficient_attention()
    df = pd.read_csv(args.prompt_path)
    
    
    generator = set_seed(args.seed)
    for idx, row in df.iterrows():
        prompt = row["Prompts"]
        clean_prompt = clean_filename(prompt)
        filename = f"{idx}_{clean_prompt}.png"
        image_path = os.path.join(args.save_dir, filename)
        # if os.path.exists(image_path):
        #     continue
        best_reward = -100

        shape = (args.population_size_schedule[0], 4, args.image_size // 8, args.image_size // 8)
        init_latents = torch.randn(shape, generator=generator, dtype=dtype).to(device)
       
        ## Test-Time Scaling
        best_image = pipe.generate(
                prompt=[prompt]*init_latents.shape[0],
                num_inference_steps=args.infer_step,
                guidance_scale=args.cfg,
                generator=generator,
                evolution_schedule=args.evolution_schedule,
                population_size_schedule=args.population_size_schedule,
                mutation_rate=args.mutation_rate,
                eta=1.0,
                guidance_reward=args.guidance_reward,
                elite_size=args.elite_size,
                iterations=args.iterations,
                evol_batch_size=init_latents.shape[0]//args.iterations,
                return_dict=False,
                latents=init_latents)[0]
           
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        clean_prompt = clean_filename(prompt)
        filename = f"{idx}_{clean_prompt}.png"
        image_path = os.path.join(args.save_dir, filename)
        best_image.save(image_path)
        
        print(f"Saved: {image_path}")
        
    print('Generation End!')
    images = load_images(args.save_dir)
    results = do_eval(
            prompt=list(df["Prompts"]), images=images, metrics_to_compute=['Aesthetic','ImageReward','HumanPreference',"Clip-Score-only"]
        )
    with open(os.path.join(args.save_dir,'results.json'), 'w+', encoding='utf-8') as f:
            json.dump(results, f)
    print('Evaluation End!')