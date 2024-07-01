# Neural STL Controller for Autonomous Visual Navigation

## Overview
This code modifies the repositories for [NoMaD](https://github.com/robodhruv/visualnav-transformer/) and [STLCG](https://github.com/StanfordASL/stlcg/). 

## Setup
It is recomemnded to build the docker image from the Dockerfile and run it. 
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
5. Run "/scripts/train_nomad.sh".
