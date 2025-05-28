from matplotlib.pyplot import flag
import torch
import os
import pandas as pd
import numpy as np
import argparse
from datetime import datetime
import json
import sys
import math
import importlib
from itertools import chain
from pathlib import Path
from VideoReward.score import *
import imageio.v2 as imageio
import io
import json
from Wan21 import wan
from Wan21.wan.configs import WAN_CONFIGS
from Wan21.wan.utils.utils import cache_video
import heapq
def get_args():
    parser = argparse.ArgumentParser(description="EvoSearch")

    parser.add_argument('--cfg', type=float, default=7.5, help='guidance for denoising process')
    parser.add_argument('--infer_step', type=int, default=50, help='total inference timestep T')
    parser.add_argument('--seed', type=int, default=42, help='Random seed to determine the initial latent.')
    parser.add_argument('--device', type=str, default='cuda', help='Device where the model inference is performed.')
    parser.add_argument('--model_name', type=str, default='Wan-AI/Wan2.1-T2V-1.3B-Diffusers', help='pre-trained model name')
    parser.add_argument('--model_path', type=str, default='/m2v_intern/hehaoran/rfpp/video_search/Wan2.1-T2V-1.3B', help='pre-trained model name')
    parser.add_argument('--prompt_path', type=str, default='./DrawBench_Prompts.csv', help='prompt file path')
    parser.add_argument('--save_dir', type=str, default='./evol-test', help='Path to save the generated images.')
    parser.add_argument('--dimension', type=str, default='composition', help='Path to save the generated images.')
    ### EvoSearch
    parser.add_argument('--evolution_schedule', nargs='+', type=int,help='Path to save the generated images.')
    parser.add_argument('--population_size_schedule', nargs='+', type=int, help='Path to save the generated images.')
    parser.add_argument('--guidance_reward', default='ImageReward',type=str,help='Guidance reward function used for search')
    parser.add_argument('--iterations', type=int, default=1, help='Iterations for doing evosearch over a batch of data')
    parser.add_argument('--number_of_N', type=int, default=100, help='scaling size')
    parser.add_argument('--mutation_rate', type=float, default=0.2, help='Mutation rate')
    parser.add_argument('--elite_size', type=int, default=2, help='Number of elites to keep across each generation')
    parser.add_argument('--height', type=int, default=832, help='frame height')
    parser.add_argument('--width', type=int, default=480, help='frame width')
    args = parser.parse_args()
    return args



def set_seed(random_seed):
    np.random.seed(int(random_seed))
    torch.manual_seed(int(random_seed))
    torch.cuda.manual_seed(int(random_seed))
    torch.cuda.manual_seed_all(int(random_seed))
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    generator = torch.manual_seed(random_seed)
    return generator
    
def clean_filename(text):
    """Clean prompt text to create safe filenames"""
    return "".join(c if c.isalnum() else "_" for c in text)[:150]

if __name__ == '__main__':
    #load args
    negative_prompt = "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards"
    args = get_args()
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir, exist_ok=True)
    
    dtype = torch.float16
    device = torch.device(args.device)
    cfg = WAN_CONFIGS['t2v-1.3B']
    pipe = wan.WanT2VEvol(
            config=cfg,
            checkpoint_dir='./Wan2.1-T2V-1.3B',
            device_id=0,
        )
    print("Initializing VideoReward model...")
    load_from_pretrained = './VideoReward'
    verifier = VideoVLMRewardInference(load_from_pretrained, device=device, dtype=dtype)
    generator = set_seed(args.seed)
    file = open('./prompts.json')
    prompts = json.load(file)
   
    for idx, prompt_ in enumerate(prompts):
        dimension = prompt_["dimension"][0]
        prompt = prompt_["prompt_en"]  # Assuming CSV has a 'prompt' column
        
        shape = (pipe.vae.model.z_dim, (33 - 1) // pipe.vae_stride[0] + 1,
                        args.height // pipe.vae_stride[1],
                        args.width // pipe.vae_stride[2])
        init_latents = torch.randn((args.population_size_schedule[0],)+shape, generator=generator, dtype=dtype).to(device)
        video = pipe.generate(
                    input_prompt=prompt,
                    n_prompt=negative_prompt,
                    frame_num=33,
                    guide_scale=5.0,
                    sampling_steps=args.infer_step,
                    noise=init_latents,
                    seed=args.seed,
                    verifier=verifier,
                    number_of_N=len(init_latents),
                    sample_solver='dpm++',
                    evolution_schedule=args.evolution_schedule,
                    population_size_schedule=args.population_size_schedule,
                    mutation_rate=args.mutation_rate,
                    guidance_reward=args.guidance_reward,
                    elite_size=args.elite_size,
                    generator=generator,
                    offload_model=False,)[0]
        
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        clean_prompt = clean_filename(prompt)
        os.makedirs(f'{args.save_dir}/{dimension}', exist_ok=True)
        video_path = f'{args.save_dir}/{dimension}/{idx}_{prompt[:180]}.mp4'
        cache_video(
                tensor=video[None],
                save_file=video_path,
                fps=16,
                nrow=1,
                normalize=True,
                value_range=(-1, 1))
        print(f"Saved: {video_path}")
    print('Generation End!')