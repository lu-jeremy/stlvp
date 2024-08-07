# STLVP

## Overview
This codebase implements the neural signal temporal logic vision planner (STLVP). STLVP builds on using signal temporal logic computation graphs (STLCG) for end-to-end visual navigation in robotics, using raw observation image inputs.  

This repository uses the goal-agnostic and goal-oriented visuomotor diffusion policy found in the [NoMaD: Goal Masking Diffusion Policies for Navigation and Exploration](https://github.com/robodhruv/visualnav-transformer/) project. Ensure that the STL functions are integrated properly in the functions calling NoMaD, specifically in train_utils.py, train_eval_loop.py, and train.py.

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
