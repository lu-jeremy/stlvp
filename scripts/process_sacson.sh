HOST="/media/corallab-s1/2tbhdd1/Jeremy"
CONTAINER="/app"

# load dataset on directory directly if you don't require dependencies
# process the dataset with installed dependencies
docker run -it -v $HOST:$CONTAINER lu1008 python /app/visualnav-transformer/train/process_bags.py -d sacson -i /app/sacson -o /app/data/sacson
