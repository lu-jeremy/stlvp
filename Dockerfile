FROM nvidia/cuda:11.3.1-cudnn8-runtime-ubuntu20.04
# change `runtime` to `devel` in the above if you require doing things like building CUDA layers.

# for OpenCV installation
ENV TZ=America/New_York
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN apt update && \
    apt install -y \
        wget git net-tools vim curl build-essential zlib1g-dev libncurses5-dev libgdbm-dev libnss3-dev libssl-dev \
        libreadline-dev libffi-dev libsqlite3-dev libbz2-dev liblzma-dev && \
        apt install -y ffmpeg libsm6 libxext6 liblz4-dev && \
    apt clean && \
    rm -rf /var/lib/apt/lists/*

RUN apt update && apt install -y lsb-release && \
        sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list' && \
        curl -s https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | apt-key add - && \
        apt update && \
        DEBIAN_FRONTEND=noninteractive apt install -y ros-noetic-desktop-full && \
        echo "source /opt/ros/noetic/setup.bash" >> ~/.bashrc

SHELL ["/bin/bash", "-c"]
RUN source ~/.bashrc && \
        DEBIAN_FRONTEND=noninteractive apt install -y ros-noetic-usb-cam ros-noetic-joy python3-rosdep python3-rosinstall python3-rosinstall-generator python3-wstool build-essential && \
        rosdep init && rosdep update && \
    apt clean && \
    rm -rf /var/lib/apt/lists/*

RUN apt update && \
    apt install tmux && \
    apt clean && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /temp

# set-up python
RUN wget https://www.python.org/ftp/python/3.9.10/Python-3.9.10.tgz && \
    tar -xvf Python-3.9.10.tgz
RUN cd Python-3.9.10 && \
    ./configure --enable-optimizations && \
    make && \
    make install
RUN rm -r /temp && \
    ln -s /usr/local/bin/python3 /usr/local/bin/python && \
    ln -s /usr/local/bin/pip3 /usr/local/bin/pip

# required for dependencies
RUN pip3 install --extra-index-url https://rospypi.github.io/simple/ sensor-msgs geometry-msgs rosbag roslz4 rospkg && \
    pip3 install torch==1.10.2+cu113 torchvision==0.11.3+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html && \
    pip3 install tqdm==4.64.0 git+https://github.com/ildoonet/pytorch-gradual-warmup-lr.git opencv-python==4.6.0.66 h5py==3.6.0 wandb==0.12.18 prettytable efficientnet-pytorch warmup-scheduler diffusers==0.11.1 lmdb vit-pytorch positional-encodings requests beautifulsoup4 matplotlib numpy pyyaml && \
    pip3 install ultralytics

# ENV PATH="/root/miniconda3/bin:${PATH}"
# ARG PATH="/root/miniconda3/bin:${PATH}"
# ENV DEBIAN_FRONTEND=noninteractive
# RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
#     mkdir /root/.conda && \
#     bash Miniconda3-latest-Linux-x86_64.sh -b && \
#     rm -f Miniconda3-latest-Linux-x86_64.sh && \
#     echo "Running $(conda --version)"

# copy as late as possible to optimize caching
COPY /utils /temp
# relative to app dir
RUN python /temp/utils/delete_roscpp_loggers.py

RUN git clone https://github.com/real-stanford/diffusion_policy.git
RUN pip3 install -e diffusion_policy 

RUN pip install git+https://github.com/ChaoningZhang/MobileSAM.git ipython graphviz timm PyGithub

# RUN mkdir -p /data/sacson/huron

# change wrong dir in process_bags.py
# RUN sed -i "s|vint_train/process_data|../temp/visualnav-transformer/train/vint_train/process_data|g" visualnav-transformer/train/process_bags.py

WORKDIR /app
