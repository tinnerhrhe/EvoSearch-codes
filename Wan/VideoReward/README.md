---
base_model: Qwen/Qwen2-VL-2B-Instruct
license: apache-2.0
---


<h2 align="center"> <strong> Improving Video Generation with Human Feedback </strong> </h2>

<div align="center">
<p align="center">
   📃 <a href="https://arxiv.org/abs/2501.13918" target="_blank">[Paper]</a> • 🌐 <a href="https://gongyeliu.github.io/videoalign/" target="_blank">[Project Page]</a> • <a href="https://github.com/KwaiVGI/VideoAlign" target="_blank">[Github]</a> • 🤗<a href="https://huggingface.co/datasets/KwaiVGI/VideoGen-RewardBench" target="_blank">[VideoGen-RewardBench]</a>• 🏆<a href="https://huggingface.co/spaces/KwaiVGI/VideoGen-RewardBench" target="_blank">[ Leaderboard]</a>
</p>
</div>

## Introduction
Welcome to VideoReward, a VLM-based reward model introduced in our paper [Improving Video Generation with Human Feedback](https://arxiv.org/abs/2501.13918). VideoReward is a multi-dimensional reward model that evaluates generated videos on three critical aspects:
* Visual Quality (VQ): The clarity, aesthetics, and single-frame reasonableness.
* Motion Quality (MQ): The dynamic stability, dynamic reasonableness, naturalness, and dynamic degress.
* Text Alignment (TA): The relevance between the generated video and the text prompt.

This versatile reward model can be used for data filtering, guidance, reject sampling, DPO, and other RL methods.

<img src=https://gongyeliu.github.io/videoalign/pics/overview.png width="100%"/>

## Usage

Please refer to our [github](https://github.com/KwaiVGI/VideoAlign) for details on usage.


## Citation

If you find this project useful, please consider citing:

```bibtex
@article{liu2025improving,
      title={Improving Video Generation with Human Feedback},
      author={Jie Liu and Gongye Liu and Jiajun Liang and Ziyang Yuan and Xiaokun Liu and Mingwu Zheng and Xiele Wu and Qiulin Wang and Wenyu Qin and Menghan Xia and Xintao Wang and Xiaohong Liu and Fei Yang and Pengfei Wan and Di Zhang and Kun Gai and Yujiu Yang and Wanli Ouyang},
      journal={arXiv preprint arXiv:2501.13918},
      year={2025}
}
