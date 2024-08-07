# Neural Signal Temporal Logic Vision Planner (STLVP)

## Overview
This codebase implements the neural STLVP, which is the first framework to integrate signal temporal logic computation graphs (STLCG) for end-to-end visual navigation in robotics. STLVP reduces sample complexity in supervised learning, non-convex optimization problems and is proven to work more efficiently than just optimal policy optimization.  

Two pipelines are proposed:  
  1) The vision pipeline leverages DeepLabV3, StableDiffusion, and MobileViT to propose semantically-segmented subgoals in pixel space and subsequently satisfy STL robustness in latent space.  
  2) The trajectory pipeline constrains the diffusion model's predicted trajectories with spatio-temporal waypoint specifications.  

We use the [X-Embodiment](https://robotics-transformer-x.github.io/) collaboration dataset ([SACSoN](https://sites.google.com/view/sacson-review/home), [RECON](https://sites.google.com/view/recon-robot/dataset)) to train the policy and generate STL waypoints.  

This project incorporates NoMaD, a visuomotor diffusion policy mapping raw observation image inputs to candidate actions. Published in the ICRA'2024 conference as _[NoMaD: Goal Masking Diffusion Policies for Navigation and Exploration](https://github.com/robodhruv/visualnav-transformer/)_, NoMaD allows for joint policy learning of goal-agnostic and goal-oriented tasks.   

Ensure that the STL functions are integrated properly in the NoMaD training functions, specifically in train_utils, train_eval_loop, and train.py.  

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
