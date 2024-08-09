# Neural Signal Temporal Logic Vision Planner (STLVP)

## Overview
This codebase implements the neural STLVP, the first framework to integrate signal temporal logic computation graphs (STLCG) for end-to-end visual navigation in robotics.  

STLVP reduces sample complexity in supervised learning and non-convex minimization problems. It has also been demonstrated to perform more efficiently than traditional optimal policy optimization methods.  

### Pipelines: 
  1) The **vision pipeline** leverages [DeepLabv3](https://arxiv.org/abs/1706.05587), [StableDiffusion](https://github.com/CompVis/stable-diffusion), and [MobileViT](https://arxiv.org/abs/2110.02178) to propose semantically-segmented subgoals in pixel space and satisfy STL robustness in latent space.  
  2) The **trajectory pipeline** constrains the diffusion model's predicted trajectories with spatio-temporal waypoint specifications.  

## Acknowledgements

This project integrates the PyTorch framework for [STLCG](https://arxiv.org/abs/2008.00097) (WAFR'2020) and the goal-agnostic/oriented diffusion policy in "NoMaD: Goal Masking Diffusion Policies for Navigation and Exploration" (ICRA'2024). NoMaD's architectural details can be viewed in its [repository](https://github.com/robodhruv/visualnav-transformer/).

Ensure that the STL functions are called properly in the NoMaD training objectives, specifically in train_utils, train_eval_loop, and train.py.  

## Dataset
The [X-Embodiment](https://robotics-transformer-x.github.io/) collaboration dataset is used to train the policy and generate STL waypoints, mainly the [SACSoN](https://sites.google.com/view/sacson-review/home) and [RECON](https://sites.google.com/view/recon-robot/dataset) datasets.  

## Setup

This project was run on an NVIDIA RTX 3090 GPU. Users should have access to workstations with Ubuntu 18.04/20.04, Python 3.9+, and CUDA 11.3+.  

The models used are from the Hugging Face directory, and hence require an API key, which should be specified as an environment variable when running the container.  

Also, it is recommended to build the image from the Dockerfile provided and run the training script (assuming the Dockerfile is in the current directory):  
```
docker build -t [image_name] .
./scripts/train_nomad.sh
```

However, if you are using a conda or virtual environment, follow these steps: 

1. Clone the following repositories [NoMaD](https://github.com/robodhruv/visualnav-transformer/) and [STLCG](https://github.com/StanfordASL/stlcg/) (PyTorch) into your environment:
```
git clone git@github.com:robodhruv/visualnav-transformer.git git@github.com:StanfordASL/stlcg.git git@github.com:real-stanford/diffusion_policy.git
```

2. Install the related dependencies.
```
pip install -e diffusion_policy/ visualnav-transformer/ stlcg/
```

3. Install the dependencies listed in the requirements.txt file. [TODO]
```
pip install -r requirements.txt
```

4. Install ROS dependencies, instructions listed [here](https://wiki.ros.org/noetic/Installation/Ubuntu).
5. Run:
```
python train.py -c ~/visualnav-transformer/train/config/nomad.yaml
```
