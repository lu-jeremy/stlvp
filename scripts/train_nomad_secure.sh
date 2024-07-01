HOST="/media/corallab-s1/2tbhdd1/Jeremy/"
CONTAINER="/app"

#docker run -it -v $HOST:$CONTAINER lu1008 python /app/visualnav-transformer/train/train.py -c /app/visualnav-transformer/train/config/nomad.yaml
nvidia-docker run -it -e WANDB_API_KEY=$KEY --shm-size=40g -e NVIDIA_DRIVER_CAPABILITIES=compute,utility -e NVIDIA_VISIBLE_DEVICES=all -v $HOST:$CONTAINER lu1008 python /app/visualnav-transformer/train/train.py -c /app/visualnav-transformer/train/config/nomad.yaml 
