# altered version of: https://github.com/robodhruv/visualnav-transformer/
# added depth topic to sync up w/ im and odom data

def get_images_odom_depth(
    bag: rosbag.Bag,
    imtopics: List[str] or str,
    odomtopics: List[str] or str,
    depthtopics: str,
    img_process_func: Any,
    odom_process_func: Any,
    rate: float = 4.0,
    ang_offset: float = 0.0,
):
    """
    Get image and odom data from a bag file

    Args:
        bag (rosbag.Bag): bag file
        imtopics (list[str] or str): topic name(s) for image data
        odomtopics (list[str] or str): topic name(s) for odom data
        img_process_func (Any): function to process image data
        odom_process_func (Any): function to process odom data
        rate (float, optional): rate to sample data. Defaults to 4.0.
        ang_offset (float, optional): angle offset to add to odom data. Defaults to 0.0.
    Returns:
        img_data (list): list of PIL images
        traj_data (list): list of odom data
    """
    # check if bag has both topics
    odomtopic = None
    imtopic = None
    if type(imtopics) == str:
        imtopic = imtopics
    else:
        for imt in imtopics:
            if bag.get_message_count(imt) > 0:
                imtopic = imt
                break
    if type(odomtopics) == str:
        odomtopic = odomtopics
    else:
        for ot in odomtopics:
            if bag.get_message_count(ot) > 0:
                odomtopic = ot
                break
    # depthtopics is always str, but can add list functionality if necessary
    if type(depthtopics) == str:
      depthtopic = depthtopics
    if not (imtopic and odomtopic):
        # bag doesn't have both topics
        return None, None

    synced_imdata = []
    synced_odomdata = []
    synced_depthdata = []
    # get start time of bag in seconds
    currtime = bag.get_start_time()

    curr_imdata = None
    curr_odomdata = None

    for topic, msg, t in bag.read_messages(topics=[imtopic, odomtopic, depthtopic]):
        if topic == imtopic:
            curr_imdata = msg
        elif topic == odomtopic:
            curr_odomdata = msg
        elif topic == depthtopic:
            curr_depthdata = msg
        if (t.to_sec() - currtime) >= 1.0 / rate:
            if curr_imdata is not None and curr_odomdata is not None:
                synced_imdata.append(curr_imdata)
                synced_odomdata.append(curr_odomdata)
                synced_depthdata.append(curr_depthdata)
                currtime = t.to_sec()

    img_data = process_images(synced_imdata, img_process_func)
    traj_data = process_odom(
        synced_odomdata,
        odom_process_func,
        ang_offset=ang_offset,
    )
    depth_data = process_images(synced_depthdata, img_process_func)

    return img_data, traj_data, depth_data

im_metric = config[args.dataset_name]["imtopics"]
depth_metric = '/depth_spherical_image/compressed'
lidar_metric = '/laserscan'

img_subdir = 'img'
depth_subdir = 'depth'
dataset_mode = 'sacson'

dataset_dir = '../../dataset'
data_to_process = '../../sacson/huron'

# added custom depth metric to im and odom processing

def main(args: argparse.Namespace):

    # load the config file
    with open("vint_train/process_data/process_bags_config.yaml", "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    # create output dir if it doesn't exist
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # iterate recursively through all the folders and get the path of files with .bag extension in the args.input_dir
    bag_files = []
    for root, dirs, files in os.walk(args.input_dir):
        for file in files:
            if file.endswith(".bag"):
                bag_files.append(os.path.join(root, file))
    if args.num_trajs >= 0:
        bag_files = bag_files[: args.num_trajs]

    print('Dataset info:', rosbag.Bag(bag_files[0]).get_type_and_topic_info())

    # processing loop
    for bag_path in tqdm.tqdm(bag_files, desc="Bags processed"):
        try:
            b = rosbag.Bag(bag_path)
        except rosbag.ROSBagException as e:
            print(e)
            print(f"Error loading {bag_path}. Skipping...")
            continue

        # name is that folders separated by _ and then the last part of the path
        traj_name = "_".join(bag_path.split("/")[-2:])[:-4]
        # print(traj_name)

        # retrieve est depth from spherical camera
        # print(b.get_type_and_topic_info()._asdict()['topics'].keys())
        # print(b.get_type_and_topic_info().topics[depth_metric])

        # synced_lidar_data = []
        # for topic, msg, t in b.read_messages(topics=[lidar_metric]):
        #   synced_lidar_data.append(msg)
        #   print(msg)
        #   break

        # load the hdf5 file
        bag_img_data, bag_traj_data, bag_depth_data = get_images_odom_depth(
            b,
            im_metric,
            config[args.dataset_name]["odomtopics"],
            depth_metric,
            eval(config[args.dataset_name]["img_process_func"]),
            eval(config[args.dataset_name]["odom_process_func"]),
            rate=args.sample_rate,
            ang_offset=config[args.dataset_name]["ang_offset"],
        )

        # print(bag_depth_data)

        if bag_img_data is None or bag_traj_data is None or bag_depth_data is None:
            print(
                f"{bag_path} did not have the topics we were looking for. Skipping..."
            )
            continue
        # remove backwards movement
        cut_trajs = filter_backwards(bag_img_data, bag_traj_data)

        for i, (img_data_i, traj_data_i) in enumerate(cut_trajs):
            traj_name_i = traj_name + f"_{i}"
            traj_folder_i = os.path.join(args.output_dir, traj_name_i)
            # make a folder for the traj
            if not os.path.exists(traj_folder_i):
                os.makedirs(traj_folder_i)
            if not os.path.exists(os.path.join(traj_folder_i, img_subdir)):
                os.makedirs(os.path.join(traj_folder_i, img_subdir))
            with open(os.path.join(traj_folder_i, "traj_data.pkl"), "wb") as f:
                pickle.dump(traj_data_i, f)
            # save the image data to disk
            for i, img in enumerate(img_data_i):
                img.save(os.path.join(traj_folder_i, img_subdir, f"{i}.jpg"))

        cut_trajs = filter_backwards(bag_depth_data, bag_traj_data)
        for i, (depth_data_i, traj_data_i) in enumerate(cut_trajs):
            traj_name_i = traj_name + f"_{i}"
            traj_folder_i = os.path.join(args.output_dir, traj_name_i)
            # make a folder for the traj
            if not os.path.exists(os.path.join(traj_folder_i, depth_subdir)):
                os.makedirs(os.path.join(traj_folder_i, depth_subdir))
            # with open(os.path.join(traj_folder_i, "traj_data.pkl"), "wb") as f:
            #     pickle.dump(traj_data_i, f)
            # # save the image data to disk

            for i, depth in enumerate(depth_data_i):
                depth.save(os.path.join(traj_folder_i, depth_subdir, f"{i}.jpg"))



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # get arguments for the recon input dir and the output dir
    # add dataset name
    parser.add_argument(
        "--dataset-name",
        "-d",
        type=str,
        help="name of the dataset (must be in process_config.yaml)",
        default="tartan_drive",
        required=True,
    )
    parser.add_argument(
        "--input-dir",
        "-i",
        type=str,
        help="path of the datasets with rosbags",
        required=True,
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        default="../datasets/tartan_drive/",
        type=str,
        help="path for processed dataset (default: ../datasets/tartan_drive/)",
    )
    # number of trajs to process
    parser.add_argument(
        "--num-trajs",
        "-n",
        default=-1,
        type=int,
        help="number of bags to process (default: -1, all)",
    )
    # sampling rate
    parser.add_argument(
        "--sample-rate",
        "-s",
        default=4.0,
        type=float,
        help="sampling rate (default: 4.0 hz)",
    )

    #args = parser.parse_args(args=['-d', dataset_mode, '-i', data_to_process, '-o', dataset_dir])

    # all caps for the dataset name
    print(f"STARTING PROCESSING {args.dataset_name.upper()} DATASET")
    main(args)
    print(f"FINISHED PROCESSING {args.dataset_name.upper()} DATASET")

