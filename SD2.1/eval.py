import torch
import os
import glob
from PIL import Image
from diffusers import DDIMScheduler, DDIMInverseScheduler
from diffusers import StableDiffusionPipeline
import pandas as pd
import numpy as np
import argparse
from datetime import datetime
from utils import do_eval
import json
# torch.set_default_device('cuda:1')
# torch.cuda.set_device('cuda:1')
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda', help='Device where the model inference is performed.')
    parser.add_argument('--prompt_path', type=str, default='./DrawBench_Prompts.csv', help='prompt file path')
    parser.add_argument('--eval_dir', type=str, default='./evo', help='Path to save the generated images.')

    args = parser.parse_args()
    return args




def load_images(image_dir, valid_extensions=('png', 'jpg', 'jpeg')):
    """
   
        image_dir: 
        valid_extensions: 
 
        List[Image.Image]:
    """
    if not os.path.isdir(image_dir):
        raise FileNotFoundError(f"dir does not exist: {image_dir}")
    
   
    image_paths = []
    for ext in valid_extensions:
        image_paths.extend(
            glob.glob(os.path.join(image_dir, f'*.{ext}'), recursive=False)
        )
        image_paths.extend(
            glob.glob(os.path.join(image_dir, f'*.{ext.upper()}'), recursive=False)
        )
    
    image_paths = sorted(list(set(image_paths)), key=lambda x: int(x.split('/')[-1].split('_')[0]))
    images = []
    for img_path in image_paths:
        try:
            with Image.open(img_path).convert("RGB") as img:
                images.append(img.copy())  
        except Exception as e:
            print(f"error {img_path}: {str(e)}")
    
    return images
    
def clean_filename(text):
    """Clean prompt text to create safe filenames"""
    return "".join(c if c.isalnum() else "_" for c in text)[:150]
if __name__ == '__main__':
    #load args
    args = get_args()
    #load model
    dtype = torch.float16
    device = torch.device(args.device)
    df = pd.read_csv(args.prompt_path)
    images = load_images(args.eval_dir)
    results = do_eval(
        prompt=list(df["Prompts"]), images=images, metrics_to_compute=['Aesthetic','ImageReward','HumanPreference',"Clip-Score-only"]
        )
    with open(os.path.join(args.eval_dir,'results.json'), 'w+', encoding='utf-8') as f:
            json.dump(results, f)