import numpy as np
import os
import pickle
import yaml
from typing import Any, Dict, List, Optional, Tuple
import tqdm
import io
import lmdb
import sys

import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF

# this is imported from the other module, so sys path insert is not necessary
from vint_train.data.data_utils import (
    img_path_to_data,
    calculate_sin_cos,
    get_data_path,
    to_local_coords,
)

# necessary for stl generation
from stl_traj_pipeline import *
from text_utils import *  # filter segmentation predictions
import stl2vecl2


"""
TODO:
- see if tweaking the threshold does anything, look into normalization
- retrieve action stats dataset config variable
- add attribute for size (optional)
- we can change the measure size for our trajectory

- q: why does stlcg just do a forward pass with always on the robustness trace for the first stl formula given the inputs, 
while we just run the rnn cell on the trace directly? does this have to do with reversed signals?
- 
"""


class ViNT_Dataset(Dataset):
    """
    Same implementation as:
    https://github.com/robodhruv/visualnav-transformer/blob/7b5b24cf12d0989fb5b5ff378d5630dd737eec3b/train/vint_train/data/vint_dataset.py
    , but with STL expression sampling capabilities.
    """
    def __init__(
            self,
            data_folder: str,
            data_split_folder: str,
            dataset_name: str,
            image_size: Tuple[int, int],
            waypoint_spacing: int,
            min_dist_cat: int,
            max_dist_cat: int,
            min_action_distance: int,
            max_action_distance: int,
            negative_mining: bool,
            len_traj_pred: int,
            learn_angle: bool,
            context_size: int,
            context_type: str = "temporal",
            end_slack: int = 0,
            goals_per_obs: int = 1,
            normalize: bool = True,
            obs_type: str = "image",
            goal_type: str = "image",
    ):
        """
        Main ViNT dataset class

        Args:
            data_folder (string): Directory with all the image data
            data_split_folder (string): Directory with filepaths.txt, a list of all trajectory names in the dataset split that are each seperated by a newline
            dataset_name (string): Name of the dataset [recon, go_stanford, scand, tartandrive, etc.]
            waypoint_spacing (int): Spacing between waypoints
            min_dist_cat (int): Minimum distance category to use
            max_dist_cat (int): Maximum distance category to use
            negative_mining (bool): Whether to use negative mining from the ViNG paper (Shah et al.) (https://arxiv.org/abs/2012.09812)
            len_traj_pred (int): Length of trajectory of waypoints to predict if this is an action dataset
            learn_angle (bool): Whether to learn the yaw of the robot at each predicted waypoint if this is an action dataset
            context_size (int): Number of previous observations to use as context
            context_type (str): Whether to use temporal, randomized, or randomized temporal context
            end_slack (int): Number of timesteps to ignore at the end of the trajectory
            goals_per_obs (int): Number of goals to sample per observation
            normalize (bool): Whether to normalize the distances or actions
            goal_type (str): What data type to use for the goal. The only one supported is "image" for now.
        """
        self.data_folder = data_folder
        self.data_split_folder = data_split_folder
        self.dataset_name = dataset_name

        traj_names_file = os.path.join(data_split_folder, "traj_names.txt")
        with open(traj_names_file, "r") as f:
            file_lines = f.read()
            self.traj_names = file_lines.split("\n")
        if "" in self.traj_names:
            self.traj_names.remove("")

        self.image_size = image_size
        self.waypoint_spacing = waypoint_spacing
        self.distance_categories = list(
            range(min_dist_cat, max_dist_cat + 1, self.waypoint_spacing)
        )
        self.min_dist_cat = self.distance_categories[0]
        self.max_dist_cat = self.distance_categories[-1]
        self.negative_mining = negative_mining
        if self.negative_mining:
            self.distance_categories.append(-1)
        self.len_traj_pred = len_traj_pred
        self.learn_angle = learn_angle

        self.min_action_distance = min_action_distance
        self.max_action_distance = max_action_distance

        self.context_size = context_size
        assert context_type in {
            "temporal",
            "randomized",
            "randomized_temporal",
        }, "context_type must be one of temporal, randomized, randomized_temporal"
        self.context_type = context_type
        self.end_slack = end_slack
        self.goals_per_obs = goals_per_obs
        self.normalize = normalize
        self.obs_type = obs_type
        self.goal_type = goal_type

        data_config_dir = os.path.join(os.getcwd(), "visualnav-transformer/train/vint_train/data")

        # load data/data_config.yaml
        with open(
                os.path.join(data_config_dir, "data_config.yaml"), "r"
        ) as f:
            all_data_config = yaml.safe_load(f)
        assert (
                self.dataset_name in all_data_config
        ), f"Dataset {self.dataset_name} not found in data_config.yaml"
        dataset_names = list(all_data_config.keys())
        dataset_names.sort()
        # use this index to retrieve the dataset name from the data_config.yaml
        self.dataset_index = dataset_names.index(self.dataset_name)
        self.data_config = all_data_config[self.dataset_name]
        self.trajectory_cache = {}
        self._load_index()
        self._build_caches()

        if self.learn_angle:
            self.num_action_params = 3
        else:
            self.num_action_params = 2

        # generate STL before dataset, then sample the cache during training
        self.stl2vecl2 = stl2vecl2.STL2VecL2(self.num_action_params, len_traj_pred)
        # maps each traj name to stl vector
        self._generate_stl()

    def _process_trajectory_bag(self, trajectories: Dict) -> np.ndarray:
        """
        Whereas trajectories would be truncated to the fixed horizon size, this function rotates and normalizes
        each given trajectory.

        Returns:
            np.ndarray: the processed trajectories.
        """

        # condition: position and yaw will always be the same size
        pos = trajectories["position"]
        yaw = trajectories["yaw"]

        if len(yaw) != len(pos) + 1:
            repeat_len = len(pos) + 1 - yaw.shape[0]
            yaw = np.concatenate([yaw, np.repeat(yaw[-1], repeat_len)])
            pos = np.concatenate([pos, np.repeat(pos[-1][None], repeat_len, axis=0)], axis=0)

        init_yaw = yaw[0]

        # transform under the initial angle, no translation
        rot_mat = np.array([
            [np.cos(init_yaw), -np.sin(init_yaw), 0],
            [np.sin(init_yaw), np.cos(init_yaw), 0],
            [0, 0, 1],
        ])

        if pos.shape[-1] == 2:
            rot_mat = rot_mat[:2, :2]

        relative_pos = pos - pos[0]
        assert type(relative_pos) == np.ndarray

        # rotate based on the relative pos
        pos = relative_pos.dot(rot_mat)[1:]

        # convert meters to normalized coordinates across dataset params
        pos /= self.data_config["metric_waypoint_spacing"] * self.waypoint_spacing

        return pos

    def _generate_stl(self):
        """
        Saves STL expressions to .pt files. Stores it in the stl_cache attribute.

        Args:
            dataset_name: the current dataset name.
            traj_runs: a list of all the trajectory runs in the dataset.
            seg_model: segmentation model for extracting landmarks.
        """

        device = "cuda:0"
        seg_model, _, _ = load_models(device=device)  # assume cuda for now

        file_to_check = os.path.join(self.data_folder, self.traj_names[0], f"stl_{self.traj_names[0]}.pt")

        if os.path.isfile(file_to_check):  # TODO: temp remove
            os.remove(file_to_check)

        # if the first trajectory's stl file is not there, generate stl for all
        if not os.path.isfile(file_to_check):
            img_extension = ".jpg"

            # DeepLabV3 preprocessing config
            preprocess = transforms.Compose([
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

            # data loader for storing stl
            tqdm_loader = enumerate(tqdm.tqdm(
                self.traj_names,
                dynamic_ncols=True,
                position=0,
                leave=True,
                desc=f"Generating STL in {self.data_folder}...",
            ))

            # for each run in the dataset
            for i, traj_name in tqdm_loader:
                # current trajetory for loading
                trajectory_path = os.path.join(self.data_folder, traj_name)
                # # for saving
                # stl_file = os.path.join(trajectory_path, f"stl_{traj_name}.pt")

                # store all the times (from img filenames)
                times = [p[:p.find(img_extension)] for p in os.listdir(trajectory_path) if p.endswith(img_extension)]
                # load the images from the trajectory
                images = torch.stack([preprocess(self._load_image(traj_name, t)) for t in times])
                images = torch.as_tensor(images, dtype=torch.float32).to(device)

                print(f"number of images/trajs: {len(images)}")

                # process images first, store STL expressions
                with torch.no_grad():
                    # TODO: check that hopefully there is no memory allocation issue here
                    outputs = seg_model(images)

                # retrieve position from trajectories
                gt_trajs = self._get_trajectory(traj_name)

                # pre-processed positions to filter
                pos = self._process_trajectory_bag(gt_trajs)

                # preprocess the ground truth trajectories according to the dataset parameters
                # f_curr, start, max_goal_dist = self.index_to_data[i]
                # _, end, _ = self._sample_goal(f_curr, start, max_goal_dist)
                # we do not need the goal position here
                # gt_trajs, _ = self._compute_actions(gt_trajs, start, end)

                # each run should have waypoints
                _, waypoints, _ = filter_preds(outputs, pos)
                # trajectory waypoint lengths are inhomogeneous
                # TODO: check that this should be (8,2) for each waypoints
                waypoints = [torch.from_numpy(w.mean(axis=0)) for w in waypoints]

                if len(waypoints) == 0: print(f"there are no landmarks in run {traj_name}")
                try:
                    # waypoints should only have 1 path
                    assert waypoints[0].ndim == 2
                except:
                    # if there aren't any waypoints, all positions are waypoints
                    print(f"num waypoints: {len(waypoints)}; small dataset? length: {len(images)}")
                    waypoints = pos

                # stl expression based on the number of waypoints
                stl_out = self.stl2vecl2.create_stl_landmarks(waypoints)

                # each trajectory will correspond with the current stl expression
                self.stl2vecl2.set_cache(traj_name, stl_out)

                # don't store stl for now

                # # write STl expressions to each trajectory
                # if not os.path.isfile(stl_file):
                #     torch.save(stl_out, stl_file)

                # print(f"successfully saved stl!")
                # print(self.stl2vecl2.get_cache())
        else:
            # load trajectories
            for traj_name in tqdm.tqdm(self.traj_names):
                # for loading
                trajectory_path = os.path.join(self.data_folder, traj_name)
                stl_file = os.path.join(trajectory_path, f"stl_{traj_name}.pt")

                # load and store stl2vec
                stl_in = torch.load(stl_file)
                self.stl2vecl2.set_cache(traj_name, stl_in)

        if len(self.stl2vecl2.get_cache()) != 0: print("stl dict is in memory!")

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_image_cache"] = None
        return state

    def __setstate__(self, state):
        self.__dict__ = state
        self._build_caches()

    def _build_caches(self, use_tqdm: bool = True):
        """
        Build a cache of images for faster loading using LMDB
        """
        cache_filename = os.path.join(
            self.data_split_folder,
            f"dataset_{self.dataset_name}.lmdb",
        )

        # Load all the trajectories into memory. These should already be loaded, but just in case.
        for traj_name in self.traj_names:
            self._get_trajectory(traj_name)

        """
        If the cache file doesn't exist, create it by iterating through the dataset and writing each image to the cache
        """
        if not os.path.exists(cache_filename):
            tqdm_iterator = tqdm.tqdm(
                self.goals_index,
                disable=not use_tqdm,
                dynamic_ncols=True,
                desc=f"Building LMDB cache for {self.dataset_name}"
            )
            with lmdb.open(cache_filename, map_size=2 ** 40) as image_cache:
                with image_cache.begin(write=True) as txn:
                    for traj_name, time in tqdm_iterator:
                        image_path = get_data_path(self.data_folder, traj_name, time)
                        with open(image_path, "rb") as f:
                            txn.put(image_path.encode(), f.read())

        # Reopen the cache file in read-only mode
        self._image_cache: lmdb.Environment = lmdb.open(cache_filename, readonly=True)

    def _build_index(self, use_tqdm: bool = False):
        """
        Build an index consisting of tuples (trajectory name, time, max goal distance)
        """
        samples_index = []
        goals_index = []

        for traj_name in tqdm.tqdm(self.traj_names, disable=not use_tqdm, dynamic_ncols=True):
            traj_data = self._get_trajectory(traj_name)
            traj_len = len(traj_data["position"])

            for goal_time in range(0, traj_len):
                goals_index.append((traj_name, goal_time))

            begin_time = self.context_size * self.waypoint_spacing
            end_time = traj_len - self.end_slack - self.len_traj_pred * self.waypoint_spacing
            for curr_time in range(begin_time, end_time):
                max_goal_distance = min(self.max_dist_cat * self.waypoint_spacing, traj_len - curr_time - 1)
                samples_index.append((traj_name, curr_time, max_goal_distance))

        return samples_index, goals_index

    def _sample_goal(self, trajectory_name, curr_time, max_goal_dist):
        """
        Sample a goal from the future in the same trajectory.
        Returns: (trajectory_name, goal_time, goal_is_negative)
        """
        goal_offset = np.random.randint(0, max_goal_dist + 1)
        if goal_offset == 0:
            trajectory_name, goal_time = self._sample_negative()
            return trajectory_name, goal_time, True
        else:
            goal_time = curr_time + int(goal_offset * self.waypoint_spacing)
            return trajectory_name, goal_time, False

    def _sample_negative(self):
        """
        Sample a goal from a (likely) different trajectory.
        """
        return self.goals_index[np.random.randint(0, len(self.goals_index))]

    def _load_index(self) -> None:
        """
        Generates a list of tuples of (obs_traj_name, goal_traj_name, obs_time, goal_time) for each observation in the dataset
        """
        index_to_data_path = os.path.join(
            self.data_split_folder,
            f"dataset_dist_{self.min_dist_cat}_to_{self.max_dist_cat}_context_{self.context_type}_n{self.context_size}_slack_{self.end_slack}.pkl",
        )
        try:
            # load the index_to_data if it already exists (to save time)
            with open(index_to_data_path, "rb") as f:
                self.index_to_data, self.goals_index = pickle.load(f)

                # ('jackal_2019-12-21-14-36-13_2_r02', 57, 20), ...,
                # print(f"self.index_to_data: {self.index_to_data}")
        except:
            # if the index_to_data file doesn't exist, create it
            self.index_to_data, self.goals_index = self._build_index()
            with open(index_to_data_path, "wb") as f:
                pickle.dump((self.index_to_data, self.goals_index), f)

    def _load_image(self, trajectory_name, time):
        image_path = get_data_path(self.data_folder, trajectory_name, time)

        try:
            with self._image_cache.begin() as txn:
                image_buffer = txn.get(image_path.encode())
                image_bytes = bytes(image_buffer)
            image_bytes = io.BytesIO(image_bytes)
            return img_path_to_data(image_bytes, self.image_size)
        except TypeError:
            print(f"Failed to load image {image_path}")

    def _compute_actions(self, traj_data, curr_time, goal_time):
        start_index = curr_time
        end_index = curr_time + self.len_traj_pred * self.waypoint_spacing + 1
        yaw = traj_data["yaw"][start_index:end_index:self.waypoint_spacing]
        positions = traj_data["position"][start_index:end_index:self.waypoint_spacing]
        goal_pos = traj_data["position"][min(goal_time, len(traj_data["position"]) - 1)]

        if len(yaw.shape) == 2:
            yaw = yaw.squeeze(1)

        if yaw.shape != (self.len_traj_pred + 1,):
            const_len = self.len_traj_pred + 1 - yaw.shape[0]
            yaw = np.concatenate([yaw, np.repeat(yaw[-1], const_len)])
            positions = np.concatenate([positions, np.repeat(positions[-1][None], const_len, axis=0)], axis=0)

        assert yaw.shape == (self.len_traj_pred + 1,), f"{yaw.shape} and {(self.len_traj_pred + 1,)} should be equal"
        assert positions.shape == (
        self.len_traj_pred + 1, 2), f"{positions.shape} and {(self.len_traj_pred + 1, 2)} should be equal"

        waypoints = to_local_coords(positions, positions[0], yaw[0])
        goal_pos = to_local_coords(goal_pos, positions[0], yaw[0])

        assert waypoints.shape == (
        self.len_traj_pred + 1, 2), f"{waypoints.shape} and {(self.len_traj_pred + 1, 2)} should be equal"

        if self.learn_angle:
            yaw = yaw[1:] - yaw[0]
            actions = np.concatenate([waypoints[1:], yaw[:, None]], axis=-1)
        else:
            actions = waypoints[1:]

        if self.normalize:
            actions[:, :2] /= self.data_config["metric_waypoint_spacing"] * self.waypoint_spacing
            goal_pos /= self.data_config["metric_waypoint_spacing"] * self.waypoint_spacing

        assert actions.shape == (self.len_traj_pred,
                                 self.num_action_params), f"{actions.shape} and {(self.len_traj_pred, self.num_action_params)} should be equal"

        return actions, goal_pos

    def _get_trajectory(self, trajectory_name):
        if trajectory_name in self.trajectory_cache:
            return self.trajectory_cache[trajectory_name]
        else:
            with open(os.path.join(self.data_folder, trajectory_name, "traj_data.pkl"), "rb") as f:
                traj_data = pickle.load(f)
            self.trajectory_cache[trajectory_name] = traj_data
            return traj_data

    def __len__(self) -> int:
        return len(self.index_to_data)

    def __getitem__(self, i: int) -> Tuple[torch.Tensor]:
        """
        Args:
            i (int): index to ith datapoint
        Returns:
            Tuple of tensors containing the context, observation, goal, transformed context, transformed observation, transformed goal, distance label, and action label
                obs_image (torch.Tensor): tensor of shape [3, H, W] containing the image of the robot's observation
                goal_image (torch.Tensor): tensor of shape [3, H, W] containing the subgoal image
                dist_label (torch.Tensor): tensor of shape (1,) containing the distance labels from the observation to the goal
                action_label (torch.Tensor): tensor of shape (5, 2) or (5, 4) (if training with angle) containing the action labels from the observation to the goal
                which_dataset (torch.Tensor): index of the datapoint in the dataset [for identifying the dataset for visualization when using multiple datasets]
        """
        f_curr, curr_time, max_goal_dist = self.index_to_data[i]
        f_goal, goal_time, goal_is_negative = self._sample_goal(f_curr, curr_time, max_goal_dist)

        # Load images
        context = []
        if self.context_type == "temporal":
            # sample the last self.context_size times from interval [0, curr_time)
            context_times = list(
                range(
                    curr_time + -self.context_size * self.waypoint_spacing,
                    curr_time + 1,
                    self.waypoint_spacing,
                )
            )
            context = [(f_curr, t) for t in context_times]
        else:
            raise ValueError(f"Invalid context type {self.context_type}")

        obs_image = torch.cat([
            self._load_image(f, t) for f, t in context
        ])

        # Load goal image
        goal_image = self._load_image(f_goal, goal_time)

        # Load other trajectory data
        curr_traj_data = self._get_trajectory(f_curr)
        curr_traj_len = len(curr_traj_data["position"])
        assert curr_time < curr_traj_len, f"{curr_time} and {curr_traj_len}"

        goal_traj_data = self._get_trajectory(f_goal)
        goal_traj_len = len(goal_traj_data["position"])
        assert goal_time < goal_traj_len, f"{goal_time} an {goal_traj_len}"

        # Compute actions
        actions, goal_pos = self._compute_actions(curr_traj_data, curr_time, goal_time)

        # Compute distances
        if goal_is_negative:
            distance = self.max_dist_cat
        else:
            distance = (goal_time - curr_time) // self.waypoint_spacing
            assert (
                               goal_time - curr_time) % self.waypoint_spacing == 0, f"{goal_time} and {curr_time} should be separated by an integer multiple of {self.waypoint_spacing}"

        actions_torch = torch.as_tensor(actions, dtype=torch.float32)
        if self.learn_angle:
            actions_torch = calculate_sin_cos(actions_torch)

        action_mask = (
                (distance < self.max_action_distance) and
                (distance > self.min_action_distance) and
                (not goal_is_negative)
        )

        # sample stl from the dataset folders
        # we assume that f_curr is just 1 trajectory for now
        stl_vec = self.stl2vecl2.get_cache()[f_curr]

        # run stl vector as well
        return (
            torch.as_tensor(obs_image, dtype=torch.float32),
            torch.as_tensor(goal_image, dtype=torch.float32),
            actions_torch,
            torch.as_tensor(distance, dtype=torch.int64),
            torch.as_tensor(goal_pos, dtype=torch.float32),
            torch.as_tensor(self.dataset_index, dtype=torch.int64),
            torch.as_tensor(action_mask, dtype=torch.float32),
            torch.as_tensor(stl_vec, dtype=torch.float32),
        )