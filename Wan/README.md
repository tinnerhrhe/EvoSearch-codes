Here, we use VideoReward as the guidance reward function during search. Please download the checkpoints of VideoReward from [Huggingface](https://huggingface.co/KwaiVGI/VideoReward).

```bash
cd VideoReward
git lfs install
git clone https://huggingface.co/KwaiVGI/VideoReward
cd ..
```
 
Then please download the Wan 1.3B model for video generation.
```
pip install "huggingface_hub[cli]"
huggingface-cli download Wan-AI/Wan2.1-T2V-1.3B --local-dir ./Wan2.1-T2V-1.3B
```