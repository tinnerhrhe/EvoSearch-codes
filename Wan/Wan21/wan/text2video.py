import gc
import logging
import math
import os
import random
from statistics import variance
import sys
import types
from contextlib import contextmanager
from functools import partial

import torch
import torch.cuda.amp as amp
import torch.distributed as dist
from tqdm import tqdm



from .distributed.fsdp import shard_model
from .modules.model import WanModel
from .modules.t5 import T5EncoderModel
from .modules.vae import WanVAE
from .utils.fm_solvers import (FlowDPMSolverMultistepScheduler,
                               get_sampling_sigmas, retrieve_timesteps)
from .utils.fm_solvers_unipc import FlowUniPCMultistepScheduler


class WanT2VEvol:

    def __init__(
        self,
        config,
        checkpoint_dir,
        device_id=0,
        rank=0,
        t5_fsdp=False,
        dit_fsdp=False,
        use_usp=False,
        t5_cpu=False,
        reward_model=None,
    ):
        r"""
        Initializes the Wan text-to-video generation model components.

        Args:
            config (EasyDict):
                Object containing model parameters initialized from config.py
            checkpoint_dir (`str`):
                Path to directory containing model checkpoints
            device_id (`int`,  *optional*, defaults to 0):
                Id of target GPU device
            rank (`int`,  *optional*, defaults to 0):
                Process rank for distributed training
            t5_fsdp (`bool`, *optional*, defaults to False):
                Enable FSDP sharding for T5 model
            dit_fsdp (`bool`, *optional*, defaults to False):
                Enable FSDP sharding for DiT model
            use_usp (`bool`, *optional*, defaults to False):
                Enable distribution strategy of USP.
            t5_cpu (`bool`, *optional*, defaults to False):
                Whether to place T5 model on CPU. Only works without t5_fsdp.
        """
        
        self.device = torch.device(f"cuda:{device_id}")
        self.config = config
        self.rank = rank
        self.t5_cpu = t5_cpu

        self.num_train_timesteps = config.num_train_timesteps
        self.param_dtype = config.param_dtype

        shard_fn = partial(shard_model, device_id=device_id)
        self.text_encoder = T5EncoderModel(
            text_len=config.text_len,
            dtype=config.t5_dtype,
            device=torch.device('cpu'),
            checkpoint_path=os.path.join(checkpoint_dir, config.t5_checkpoint),
            tokenizer_path=os.path.join(checkpoint_dir, config.t5_tokenizer),
            shard_fn=shard_fn if t5_fsdp else None)

        self.vae_stride = config.vae_stride
        self.patch_size = config.patch_size
        self.vae = WanVAE(
            vae_pth=os.path.join(checkpoint_dir, config.vae_checkpoint),
            device=self.device)

        logging.info(f"Creating WanModel from {checkpoint_dir}")
        self.model = WanModel.from_pretrained(checkpoint_dir)
        self.reward_model= reward_model
        self.model.eval().requires_grad_(False)
        # self.reward_model.eval().requires_grad_(False)
        if use_usp:
            from xfuser.core.distributed import \
                get_sequence_parallel_world_size

            from .distributed.xdit_context_parallel import (usp_attn_forward,
                                                            usp_dit_forward)
            for block in self.model.blocks:
                block.self_attn.forward = types.MethodType(
                    usp_attn_forward, block.self_attn)
            self.model.forward = types.MethodType(usp_dit_forward, self.model)
            self.sp_size = get_sequence_parallel_world_size()
        else:
            self.sp_size = 1

        if dist.is_initialized():
            dist.barrier()
        if dit_fsdp:
            self.model = shard_fn(self.model)
        else:
            self.model.to(self.device)

        self.sample_neg_prompt = config.sample_neg_prompt
    def evosearch(self,
                 input_prompt,
                 ## EvoSearch args ##
                 elite_size: int = 3,
                 generation_steps: int = 0,
                 mutation_rate: float = 0.2,
                 evolution_schedule=None,
                 population_size_schedule=None,
                 verifier=None,
                 ###
                 size=(1280, 720),
                 frame_num=81,
                 shift=5.0,
                 number_of_N=1,
                 sample_solver='unipc',
                 sampling_steps=50,
                 guide_scale=5.0,
                 n_prompt="",
                 seed=-1,
                 noise=None,
                 generator=None,
                 offload_model=True):
        r"""
        Generates video frames from text prompt using diffusion process.

        Args:
            input_prompt (`str`):
                Text prompt for content generation
            size (tupele[`int`], *optional*, defaults to (1280,720)):
                Controls video resolution, (width,height).
            frame_num (`int`, *optional*, defaults to 81):
                How many frames to sample from a video. The number should be 4n+1
            shift (`float`, *optional*, defaults to 5.0):
                Noise schedule shift parameter. Affects temporal dynamics
            sample_solver (`str`, *optional*, defaults to 'unipc'):
                Solver used to sample the video.
            sampling_steps (`int`, *optional*, defaults to 40):
                Number of diffusion sampling steps. Higher values improve quality but slow generation
            guide_scale (`float`, *optional*, defaults 5.0):
                Classifier-free guidance scale. Controls prompt adherence vs. creativity
            n_prompt (`str`, *optional*, defaults to ""):
                Negative prompt for content exclusion. If not given, use `config.sample_neg_prompt`
            seed (`int`, *optional*, defaults to -1):
                Random seed for noise generation. If -1, use random seed.
            offload_model (`bool`, *optional*, defaults to True):
                If True, offloads models to CPU during generation to save VRAM

        Returns:
            torch.Tensor:
                Generated video frames tensor. Dimensions: (C, N H, W) where:
                - C: Color channels (3 for RGB)
                - N: Number of frames (81)
                - H: Frame height (from size)
                - W: Frame width from size)
        """
        # preprocess
        F = frame_num
        if noise is None:
            target_shape = (self.vae.model.z_dim, (F - 1) // self.vae_stride[0] + 1,
                        size[1] // self.vae_stride[1],
                        size[0] // self.vae_stride[2])
        else:
            target_shape = noise[0].shape
        seq_len = math.ceil((target_shape[2] * target_shape[3]) /
                            (self.patch_size[1] * self.patch_size[2]) *
                            target_shape[1] / self.sp_size) * self.sp_size

        if n_prompt == "":
            n_prompt = self.sample_neg_prompt
        

        if not self.t5_cpu:
            self.text_encoder.model.to(self.device)
            context = self.text_encoder([input_prompt], self.device)
            context_null = self.text_encoder([n_prompt], self.device)
            if offload_model:
                self.text_encoder.model.cpu()
        else:
            context = self.text_encoder([input_prompt], torch.device('cpu'))
            context_null = self.text_encoder([n_prompt], torch.device('cpu'))
            context = [t.to(self.device) for t in context]
            context_null = [t.to(self.device) for t in context_null]

        
        @contextmanager
        def noop_no_sync():
            yield

        no_sync = getattr(self.model, 'no_sync', noop_no_sync)
        
        # evaluation mode
        with amp.autocast(dtype=self.param_dtype), torch.no_grad(), no_sync():
            sample_schedulers = []
            if sample_solver == 'unipc':
                for _ in range(number_of_N):
                    sample_scheduler = FlowUniPCMultistepScheduler(
                        num_train_timesteps=self.num_train_timesteps,
                        shift=1,
                        use_dynamic_shifting=False)
                    sample_scheduler.set_timesteps(
                        sampling_steps, device=self.device, shift=shift)
                    sample_schedulers.append(sample_scheduler)
                timesteps = sample_scheduler.timesteps
            elif sample_solver == 'dpm++':
                for _ in range(number_of_N):
                    sample_scheduler = FlowDPMSolverMultistepScheduler(
                        num_train_timesteps=self.num_train_timesteps,
                        shift=1,
                        algorithm_type='sde-dpmsolver++',
                        use_dynamic_shifting=False)
                    sampling_sigmas = get_sampling_sigmas(sampling_steps, shift)
                    timesteps, _ = retrieve_timesteps(
                        sample_scheduler,
                        device=self.device,
                        sigmas=sampling_sigmas)
                    sample_schedulers.append(sample_scheduler)
            else:
                raise NotImplementedError("Unsupported solver.")

            # sample videos
            arg_c = {'context': context, 'seq_len': seq_len}
            arg_null = {'context': context_null, 'seq_len': seq_len}
            
            ## EvoSearch
            latent_to_decode_fn=lambda x: latent_to_decode(vae=self.vae, latent=x)
            generation_steps_id = generation_steps
            std_list=[-1 for _ in evolution_schedule]
            current_step = evolution_schedule[generation_steps]
            if noise is None:
                noise = [
                    torch.randn(
                    target_shape[0],
                    target_shape[1],
                    target_shape[2],
                    target_shape[3],
                    dtype=torch.float32,
                    generator=generator).to(self.device)
                        ]
            latents_total=noise
            for j, t in zip(range(current_step,sampling_steps),timesteps[current_step:]):
                latents_list = []
                for k in range(number_of_N):
                    latents = latents_total[k]
                    latent_model_input = latents[None]
                    timestep = [t]

                    timestep = torch.stack(timestep)

                    self.model.to(self.device)
                    noise_pred_cond = self.model(
                        latent_model_input, t=timestep, **arg_c)[0]
                    noise_pred_uncond = self.model(
                        latent_model_input, t=timestep, **arg_null)[0]

                    noise_pred = noise_pred_uncond + guide_scale * (
                        noise_pred_cond - noise_pred_uncond)
                   
                    latents_, _, variance, std = sample_schedulers[k].step(
                        noise_pred.unsqueeze(0),
                        t,
                        latents.unsqueeze(0),
                        return_dict=False,
                        generator=generator)
                    latents_list.append(latents_)
                    if j in evolution_schedule:
                        self.population_list[generation_steps_id].append(latents_)
                        self.variance_list[generation_steps_id].append(variance)
                latents_total = torch.cat(latents_list)
                if j in evolution_schedule:
                    std_list[generation_steps_id]=std
                    generation_steps_id += 1
            reward_list=[]
            for k in range(number_of_N):
                population_video = latent_to_decode_fn(latents_total[k])[0]
                self.video_list.append(population_video)
                reward = verifier.reward([population_video.permute(1,0,2,3)], [input_prompt], use_norm=True)
                reward_list.append(reward[0]['Overall'])
            rewards = torch.tensor(reward_list).to(self.device)
            for id in range(generation_steps,len(self.rewards_list)):
                self.rewards_list[id].append(rewards)
            population = torch.cat(self.population_list[generation_steps])
            
            mean_std=std_list[generation_steps]
            variance= torch.cat(self.variance_list[generation_steps])
            rewards = torch.cat(self.rewards_list[generation_steps])
            # print(torch.sort(rewards,descending=True)[0])
            elite_rewards= rewards
            elite_rew, elite_indices = torch.topk(elite_rewards, elite_size)
            if elite_rew[0]>self.best_reward:
                self.best_reward = elite_rew[0]
                ind = elite_indices[0]
                self.best_video = self.video_list[ind]

            elites = population[elite_indices]
            parents = []
            population_size= population_size_schedule[generation_steps+1]
            for _ in range(population_size-elite_size):
                candidates = torch.randperm(population.shape[0])[:int(population.shape[0]*0.9)]
                candidate_rewards = torch.tensor(rewards)[candidates]#rewards[candidates]
                winner = candidates[torch.argmax(candidate_rewards)]
                parents.append(population[winner])
            parents = torch.stack(parents)
            if generation_steps==0:
                children = parents * math.sqrt(1 - mutation_rate**2) + mutation_rate * torch.randn_like(parents)
            else:
                children = parents  + mean_std * torch.randn_like(parents)
            children = torch.cat([elites,children])
            return children
            
    def generate(self,
                 input_prompt,
                 ## EvoSearch args ##
                 elite_size: int = 3,
                 guidance_reward: str='VideoReward',
                 mutation_rate: float = 0.2,
                 evolution_schedule=None,
                 population_size_schedule=None,
                 verifier=None,
                 ###
                 size=(1280, 720),
                 frame_num=81,
                 shift=5.0,
                 number_of_N=1,
                 sample_solver='unipc',
                 sampling_steps=50,
                 guide_scale=5.0,
                 n_prompt="",
                 seed=-1,
                 noise=None,
                 generator=None,
                 offload_model=True):
        r"""
        Generates video frames from text prompt using diffusion process.

        Args:
            input_prompt (`str`):
                Text prompt for content generation
            size (tupele[`int`], *optional*, defaults to (1280,720)):
                Controls video resolution, (width,height).
            frame_num (`int`, *optional*, defaults to 81):
                How many frames to sample from a video. The number should be 4n+1
            shift (`float`, *optional*, defaults to 5.0):
                Noise schedule shift parameter. Affects temporal dynamics
            sample_solver (`str`, *optional*, defaults to 'unipc'):
                Solver used to sample the video.
            sampling_steps (`int`, *optional*, defaults to 40):
                Number of diffusion sampling steps. Higher values improve quality but slow generation
            guide_scale (`float`, *optional*, defaults 5.0):
                Classifier-free guidance scale. Controls prompt adherence vs. creativity
            n_prompt (`str`, *optional*, defaults to ""):
                Negative prompt for content exclusion. If not given, use `config.sample_neg_prompt`
            seed (`int`, *optional*, defaults to -1):
                Random seed for noise generation. If -1, use random seed.
            offload_model (`bool`, *optional*, defaults to True):
                If True, offloads models to CPU during generation to save VRAM

        Returns:
            torch.Tensor:
                Generated video frames tensor. Dimensions: (C, N H, W) where:
                - C: Color channels (3 for RGB)
                - N: Number of frames (81)
                - H: Frame height (from size)
                - W: Frame width from size)
        """
        # preprocess
        F = frame_num
        if noise is None:
            target_shape = (self.vae.model.z_dim, (F - 1) // self.vae_stride[0] + 1,
                        size[1] // self.vae_stride[1],
                        size[0] // self.vae_stride[2])
        else:
            target_shape = noise[0].shape
        seq_len = math.ceil((target_shape[2] * target_shape[3]) /
                            (self.patch_size[1] * self.patch_size[2]) *
                            target_shape[1] / self.sp_size) * self.sp_size

        if n_prompt == "":
            n_prompt = self.sample_neg_prompt
        

        if not self.t5_cpu:
            self.text_encoder.model.to(self.device)
            context = self.text_encoder([input_prompt], self.device)
            context_null = self.text_encoder([n_prompt], self.device)
            if offload_model:
                self.text_encoder.model.cpu()
        else:
            context = self.text_encoder([input_prompt], torch.device('cpu'))
            context_null = self.text_encoder([n_prompt], torch.device('cpu'))
            context = [t.to(self.device) for t in context]
            context_null = [t.to(self.device) for t in context_null]

        
        @contextmanager
        def noop_no_sync():
            yield

        no_sync = getattr(self.model, 'no_sync', noop_no_sync)
        
        # evaluation mode
        with amp.autocast(dtype=self.param_dtype), torch.no_grad(), no_sync():
            sample_schedulers = []
            if sample_solver == 'unipc':
                for _ in range(number_of_N):
                    sample_scheduler = FlowUniPCMultistepScheduler(
                        num_train_timesteps=self.num_train_timesteps,
                        shift=1,
                        use_dynamic_shifting=False)
                    sample_scheduler.set_timesteps(
                        sampling_steps, device=self.device, shift=shift)
                    sample_schedulers.append(sample_scheduler)
                timesteps = sample_scheduler.timesteps
            elif sample_solver == 'dpm++':
                for _ in range(number_of_N):
                    sample_scheduler = FlowDPMSolverMultistepScheduler(
                        num_train_timesteps=self.num_train_timesteps,
                        shift=1,
                        algorithm_type='sde-dpmsolver++',
                        use_dynamic_shifting=False)
                    sampling_sigmas = get_sampling_sigmas(sampling_steps, shift)
                    timesteps, _ = retrieve_timesteps(
                        sample_scheduler,
                        device=self.device,
                        sigmas=sampling_sigmas)
                    sample_schedulers.append(sample_scheduler)
            else:
                raise NotImplementedError("Unsupported solver.")

            # sample videos
            arg_c = {'context': context, 'seq_len': seq_len}
            arg_null = {'context': context_null, 'seq_len': seq_len}
            
            ## Initialization
            self.best_reward=-100
            self.best_video=None
            self.video_list=[]
            generation_steps = 0
        
            self.population_list = [[] for _ in evolution_schedule]
            self.rewards_list = [[] for _ in evolution_schedule]
            self.variance_list = [[] for _ in evolution_schedule]
            if noise is None:
                noise = [
                    torch.randn(
                    target_shape[0],
                    target_shape[1],
                    target_shape[2],
                    target_shape[3],
                    dtype=torch.float32,
                    generator=generator).to(self.device)
                        ]
            latents_total=noise
            for j, t in enumerate(tqdm(timesteps)):
                latents_list = []
                if j in evolution_schedule:
                    latents_total = self.evosearch(
                        input_prompt=input_prompt,
                        n_prompt=n_prompt,
                        frame_num=frame_num,
                        guide_scale=guide_scale,
                        noise=latents_total,
                        seed=seed,
                        number_of_N=number_of_N,
                        sample_solver=sample_solver,
                        evolution_schedule=evolution_schedule,
                        population_size_schedule=population_size_schedule,
                        elite_size=elite_size,
                        mutation_rate=mutation_rate,
                        verifier=verifier,
                        generator=generator,
                        offload_model=False,
                        )
                    generation_steps += 1
                    print('Updated best reward',self.best_reward)
                    number_of_N = len(latents_total)
                    sample_schedulers = []
                    for _ in range(number_of_N):
                        sample_scheduler = FlowDPMSolverMultistepScheduler(
                            num_train_timesteps=self.num_train_timesteps,
                            shift=1,
                            algorithm_type='sde-dpmsolver++',
                            use_dynamic_shifting=False)
                        sampling_sigmas = get_sampling_sigmas(sampling_steps, shift)
                        timesteps, _ = retrieve_timesteps(
                            sample_scheduler,
                            device=self.device,
                            sigmas=sampling_sigmas)
                        sample_schedulers.append(sample_scheduler)
                for k in range(number_of_N):
                    latents = latents_total[k]
                    latent_model_input = latents[None]
                    timestep = [t]

                    timestep = torch.stack(timestep)

                    self.model.to(self.device)
                    noise_pred_cond = self.model(
                        latent_model_input, t=timestep, **arg_c)[0]
                    noise_pred_uncond = self.model(
                        latent_model_input, t=timestep, **arg_null)[0]

                    noise_pred = noise_pred_uncond + guide_scale * (
                        noise_pred_cond - noise_pred_uncond)
                   
                    latents_, _, _, _ = sample_schedulers[k].step(
                        noise_pred.unsqueeze(0),
                        t,
                        latents.unsqueeze(0),
                        return_dict=False,
                        generator=generator)
                    latents_list.append(latents_)
                latents_total=torch.cat(latents_list)
            x0 = latents_total
            if offload_model:
                self.model.cpu()
                torch.cuda.empty_cache()
            if self.rank == 0:
                videos =[]
                for item in x0:
                    videos.append(self.vae.decode(item.unsqueeze(0))[0])
                videos=torch.stack(videos)
        with torch.no_grad():
            rewards = verifier.reward(videos.permute(0,2,1,3,4), [input_prompt]*videos.shape[0], use_norm=True)
        rewards = torch.tensor([rewards[i]['Overall'] for i in range(len(rewards))]).to(self.device)
        elite_rew, elite_indices = torch.topk(rewards, 1)
        if elite_rew[0]>self.best_reward:
            self.best_reward = elite_rew[0]
            ind = elite_indices[0]
            self.best_video = videos[ind]
        # Offload all models
        print('Updated best reward',self.best_reward)
        del noise, latents
        del sample_schedulers
        if offload_model:
            gc.collect()
            torch.cuda.synchronize()
        if dist.is_initialized():
            dist.barrier()

        return (self.best_video,)
def latent_to_decode(vae,latent):
    return vae.decode(latent[None])