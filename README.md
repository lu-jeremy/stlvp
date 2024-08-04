# A Neural Signal Temporal Logic Vision Planner

## Overview
This codebase builds on using the signal temporal logic computation graph (STLCG) platform for end-to-end visual navigation in robotics using raw observation image inputs.

Clone the following repositories and their dependencies: [NoMaD](https://github.com/robodhruv/visualnav-transformer/) and [STLCG](https://github.com/StanfordASL/stlcg/) (PyTorch).

## Setup

This project was run on an NVIDIA RTX 3090 GPU. Users should have access to workstations with Ubuntu 18.04/20.04, Python 3.9+, and CUDA 11.3+.


The models used are from the Hugging Face directory, and hence require an API key, which should be specified as an environment variable when running the container. Also, it is recommended to build the image from the Dockerfile provided and run the training script.
```
docker build -t [image_name] .
./scripts/train_nomad.sh
```

However, if you are using a conda or virtual environment, follow these steps: 

1. Clone the repositories into your environment.
```
git clone git@github.com:robodhruv/visualnav-transformer.git git@github.com:StanfordASL/stlcg.git git@github.com:real-stanford/diffusion_policy.git
```

2. Install the dependencies related to the GitHub repos.
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
