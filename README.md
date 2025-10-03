
# Scaling Image and Video Generation via Test-Time Evolutionary Search


## 🛠️ Instructions
Set up the environment
```
cd EvoSearch-codes
conda create -n evosearch python=3.9.0 -y
conda activate evosearch
pip install -r requirements.txt
```
You can try EvoSearch on Stable Diffusion 2.1 via the following commands.
```
cd SD2.1
python evosearch.py --guidance_reward ImageReward --evolution_schedule 0 20 30 40 --population_size_schedule 42 26 26 26 30
``` 
Guidance reward function is pre-defined for search, while evolution schedule and population size schedule can be freely chosen, considering the available test-time computation budget.

You can try EvoSearch on a more advanced T2I model Flux.1-dev via the following commands.
```
cd FLUX
python evosearch.py --guidance_reward ImageReward --evolution_schedule 5 20 30 40 --population_size_schedule 14 7 7 7 10
```

For video generation, please first download the guidance reward model and T2V model following the [instructions](./Wan/README.md). Then you can play EvoSearch via the following commands.
```
cd Wan
python evosearch.py --evolution_schedule 5 20 30 45 --population_size_schedule 10 5 5 5 5
```

