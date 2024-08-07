import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torchvision import transforms
from torch.utils.data import DataLoader
from torch import autocast

import numpy as np
import matplotlib.pyplot as plt
import cv2
import tqdm
from PIL import Image
import wandb

import sys
from typing import Tuple, Union, List
import os
import warnings
import time
import gc
import shutil

from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

from viz_utils import *
from text_utils import *
from constants import *

sys.path.insert(0, "visualnav-transformer/train")
from vint_train.data.data_utils import VISUALIZATION_IMAGE_SIZE
from vint_train.visualizing.visualize_utils import to_numpy, from_numpy

sys.path.insert(0, "stlcg/src")  # disambiguate path names
import stlcg
import stlviz as viz

# loading in deeplabv3 custom repo
sys.path.insert(0, "DeepLabV3Plus-Pytorch")
from network import modeling

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:21000"
sys.setrecursionlimit(10000)  # needed for STL robustness

traj_path = os.path.join(WP_DIR, TRAJ_DIR)
subgoal_dir = os.path.join(traj_path, "subgoal")
goal_dir = os.path.join(traj_path, "goal")

if not os.path.isdir(subgoal_dir):
    os.makedirs(subgoal_dir, exist_ok=True)

if not os.path.isdir(goal_dir):
    os.makedirs(goal_dir, exist_ok=True)


def load_models(device: torch.device) -> Tuple:
    """
    Loads the pre-trained ViT models onto the specified device.

    Args:
        device (`torch.device`):
            If GPU is not available, use CPU.

    Returns:
        `tuple`:
          All models used during the STL creation process. Encoder and text-to-image models are not used for waypoint
          generation.
    """
    weights_dir = os.path.join(os.getcwd(), "pretrained_weights")
    deeplab_dir = os.path.join(weights_dir, "deeplab/best_deeplabv3plus_resnet101_cityscapes_os16.pth.tar")

    num_classes = 19
    output_stride = 8

    # semantic segmentation model
    deeplab = modeling.__dict__["deeplabv3plus_resnet101"](num_classes=num_classes, output_stride=output_stride).eval()
    deeplab.load_state_dict(torch.load(deeplab_dir)["model_state"])

    return deeplab.to(device), None, None


def generate_waypoints(
        dataset: DataLoader,
        models: tuple,
        device: torch.device,
        enable_intervals: bool = ENABLE_INTERVALS,
        load_waypoints: bool = LOAD_WAYPOINTS,
        visualize_dataset: bool = VISUALIZE_DATASET,
        visualize_subgoals: bool = VISUALIZE_SUBGOALS,
        process_subgoals: bool = PROCESS_SUBGOALS,
        process_goals: bool = PROCESS_GOALS,
) -> Union[None, Tuple]:
    """
    Generates latent waypoints from the full, un-shuffled dataset.

    Args:
        dataset (`Dataloader`):
            Unshuffled dataset to load batches in.
        models (`tuple`):
            All model for segmentation, text-to-image, and encoding.
        device (`torch.device`):
            Encoder model for waypoint latents.
        enable_intervals (`bool`):
            Enable interval creation and loading.
        load_waypoints (`bool`):
            Determines whether to generate or load waypoint files.
        visualize_dataset (`bool`):
            Determines the visualization of the dataset images.
        visualize_subgoals (`bool`):
            Determines the visualization of generated subgoal images.
        process_subgoals (`bool`):
            Determines subgoal processing.
        process_goals (`bool`):
            Determines goal processing.

    Returns:
        `None` or `tuple`:
            If load_waypoints is true, only files will be generated, and no values will be returned.
            Otherwise, a tuple is returned with the first element being the WP latents and interval, and the second
            being the goal latents.
    """
    print(f"\nGenerate waypoint parameters: \n" +
          "─" * 10 + "\n" +
          f"Device: {device}\n" +
          f"# GPUs: {torch.cuda.device_count()}\n\n" +
          f"Load waypoints: {load_waypoints}\n\n" +
          f"Vis dataset: {visualize_dataset}\n" +
          f"Vis subgoals: {visualize_subgoals}\n\n" +
          f"Process subgoals: {process_subgoals}\n" +
          f"Process goals: {process_goals}\n\n" +
          f"Intervals: {enable_intervals}\n" +
          f"Persistence threshold (Δt): {PERSISTENCE_THRESH}\n" +
          "─" * 10 + "\n")

    # DeepLabV3 preprocessing config
    preprocess = transforms.Compose([
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # if the folder has no files, generate new latents
    if not load_waypoints:
        for idx, data in enumerate(tqdm.tqdm(dataset, desc="Generating waypoints...")):
            # no need to overwrite the same files, skip if detected
            subgoal_traj_file = os.path.join(subgoal_dir, f"{idx}.pt")
            goal_traj_file = os.path.join(goal_dir, f"{idx}.pt")

            if process_subgoals and os.path.isfile(subgoal_traj_file):
                print("SKIP")
                continue

            if process_goals and os.path.isfile(goal_traj_file):
                print("SKIP")
                continue

            print(f"Begin processing {idx} batch....")

            start_time = time.time()

            (
                obs_imgs,
                goal_img,
                actions,
                distance,
                goal_pos,
                dataset_idx,
                action_mask,
            ) = data

            obs_imgs = torch.split(obs_imgs, 3, dim=1)

            obs_imgs = [preprocess(img) for img in obs_imgs]
            obs_imgs = torch.stack(obs_imgs)[-1].to(device)

            # retrieve intervals based on segmentation maps
            seg_model, _, _ = models

            with torch.no_grad():
                outputs = seg_model(obs_imgs)

            _, trajs, intervals = filter_preds(outputs, actions)

            # we cannot continue process if there are no valid waypoints
            if len(intervals) == 0:
                # trajs are not saved for maps with no predictions
                print("Found no predictions, skipping...")
                continue

            assert len(trajs) == len(intervals)

            print(f"LEN TRAJ: {len(trajs)}, LEN INTERVALS: {len(intervals)}")

            # save traj and intervals to dir
            torch.save((trajs, intervals), subgoal_traj_file)
            # TODO
            # torch.save(, goal_traj_file)

            print("ELAPSED TIME: %.2f s." % (time.time() - start_time))

        # torch.save(z_batch, latent_file)
        print("Finished saving waypoint latents!")

        # program doesn't need to continue at this point
        sys.exit()
    else:
        if enable_intervals:
            # z_w = None
            # intervals = None
            w_traj = []
            intervals = []
            g_traj = []

            # subgoal takes priority, as some preds are skipped during processing
            for root, dirs, files in os.walk(subgoal_dir):
                for f in tqdm.tqdm(files, "Loading waypoints..."):
                    batch = torch.load(os.path.join(root, f))

                    assert isinstance(batch[0][0], torch.Tensor), "Traj is not a tensor"

                    w_traj.append(batch[0])
                    intervals.append(batch[1])

                    # TODO: add goals
                    # # add matching latent file with subgoal
                    # batch = torch.load(os.path.join(goal_dir, f))
                    # z_g.append(batch)

            print(f"Successfully loaded waypoint latents!\n"
                  f"# wp: {len(w_traj)}, # intervals: {len(intervals)}, # goals: {len(g_traj)}\n")

            assert len(w_traj) == len(intervals),\
                "The number of subgoal/goal latent files do not match."

            return (w_traj, intervals), None  # TODO goals


def predict_start_samples(
        noisy_action: torch.Tensor,
        noise_pred: torch.Tensor,
        timesteps: torch.IntTensor,
        noise_scheduler: DDPMScheduler,
        action_stats: dict,
        device: torch.device,
):
    """
    Predicts x_0 based on batches of noise predictions and current diffused x_t's on uniformly distributed
    timesteps.

    Returns:
        `torch.tensor`:
            The predicted, un-normalized goal-conditioned actions for the batch.
    """
    # denoise noisy action based on pred noise, TODO add this to stl loss function
    diffusion_output = []

    for k in timesteps:
        # predict x_0 through by re-parametrizing the forward process
        orig_sample = noise_scheduler.step(
            model_output=noise_pred[k],
            sample=noisy_action[k],
            timestep=k,
        ).pred_original_sample

        diffusion_output.append(orig_sample)
    # aggregate each sample in the batch
    diffusion_output = torch.stack(diffusion_output)

    # un-normalize the diffusion outputs
    action_max = torch.from_numpy(action_stats["max"]).to(device)
    action_min = torch.from_numpy(action_stats["min"]).to(device)

    diffusion_output = diffusion_output.reshape(diffusion_output.shape[0], -1, 2)
    diffusion_output = (diffusion_output + 1) / 2
    diffusion_output = diffusion_output * (action_max - action_min) + action_min
    gc_actions = torch.cumsum(diffusion_output, dim=1)

    # STL RNN cells require torch.float32
    return gc_actions.to(torch.float32)


def compute_stl_loss(
        pred_trajs: torch.Tensor,
        wp_data: torch.Tensor,
        goal_data: torch.Tensor,
        device: torch.device,
        streams: List,
        dataset_idx: int,
        action_mask: torch.Tensor,
        margin: int = 0,  # TODO: change
        visualize_stl: bool = VISUALIZE_STL,
        vis_freq: int = VIS_FREQ,
        threshold: List = SIM_THRESH,
) -> Union[None, torch.Tensor]:
    """
    Generates STL formula and inputs for robustness, while computing the STL loss based on robustness.

    Broadcast each target latent to all latent obs during similarity computation.
    (1) Broadcast target latent to element-wise multiply with obs latents.
    (2) Broadcast obs latent to each target.

    Args:
        pred_trajs (`torch.Tensor`):
            Online training observations.
        wp_data (`torch.Tensor`):
            Waypoint latents to perform similarity with + intervals.
        goal_data (`torch.Tensor`):
            Goal latents to perform similarity with.
        device (`torch.device`):
            Device to transfer tensors onto.
        models (`tuple`):
            Encoder for observation latents.
        streams (`List`):
            List of torch.cuda.Stream's instances to deploy runs on a multi-stream setup.
        margin (`int`, defaults to 0):
           Margin in the ReLU objective.
        visualize_stl (`bool`, defaults to `VISUALIZE_STL`):
            Determines the STL formula visualization.
        threshold (`List`, defaults to `SIM_THRESH`):
            Threshold values for STL similarity expressions.
    Returns:
        `torch.Tensor`:
            The computed STL loss with no loss weight.
    """
    flush()

    batch_size = 256
    if pred_trajs.size(0) < batch_size:
        print(f"NOT LARGE ENOUGH: {pred_trajs.size()}")
        return None

    start_time = time.time()

    # unpacking tuple
    wp_trajs = wp_data[0]
    intervals = wp_data[1]
    goal_trajs = goal_data

    # full expression used for training
    full_exp = None
    # signal inputs to the robustness function
    full_inputs = ()
    # used for similarity visualization
    sims = []
    annots = []

    test_start = time.time()

    # print(f"len wp_trajs: {len(wp_trajs)}, len intervals: {len(intervals)}, len pred_trajs: {len(pred_trajs)}")

    # processes a multi-stream setup for runs on parallel GPU kernels
    for i, stream in enumerate(streams):
        inner_inputs, inner_exp = process_run(
            curr_stream=stream,
            curr_run=wp_trajs[i],
            curr_interval=intervals[i],
            # curr_goal=goal_trajs[i],
            pred_trajs=pred_trajs,
            threshold=threshold,
            sims=sims,
            annots=annots,
            action_mask=action_mask,
            device=device,
        )

        # full input tuples are not concerned with individual values
        if len(full_inputs) == 0:
            full_inputs = inner_inputs
        else:
            full_inputs = (full_inputs, inner_inputs)

        # recursively add inner exp to full exp
        if full_exp is None:
            full_exp = inner_exp
        else:
            full_exp |= inner_exp

    for stream in streams:
        stream.synchronize()

    if VISUALIZE_SIM:
        if dataset_idx % vis_freq == 0:
            # sims = [sim.mean() for sim in sims[:20]]
            sims = sims[:20]
            annots = annots[:20]

            fig, ax = plt.subplots()
            x = range(0, len(sims) * 100, 100)
            ax.plot(x, sims, marker='o', linestyle='-', label='Similarities')
            # ax.plot(x[:-1], sims[:-1], marker='o', linestyle='-', label='Similarities')
            # ax.plot(x[-1], sims[-1], marker='o', linestyle='-', color="red", label='Similarities')

            for i, a in enumerate(annots):
                ax.annotate(a, (x[i], sims[i]))

            ax.set_xlabel("Time")
            ax.set_ylabel("Similarity Metrics")
            plt.title(f"{len(wp_trajs)} Runs, 1 Training Iteration")
            plt.savefig(os.path.join(IMG_DIR, f"run_{dataset_idx}.png"))

        print(f"COSSIM RECURSIVE Time (s): {time.time() - test_start}")

    if full_exp is None:
        raise ValueError(f"Formula is not properly defined.")

    # saves a digraph of the STL formula
    if visualize_stl:
        digraph = viz.make_stl_graph(full_exp)
        viz.save_graph(digraph, "utils/formula")
        print("Saved formula CG successfully.")

    # recursive depth may result in overloading stack memory
    robustness = (-full_exp.robustness(full_inputs)).squeeze()
    stl_loss = torch.relu(robustness - margin).mean()

    print(f"STL loss: {stl_loss}, \tSTL Computation Time (s): {time.time() - start_time}")

    flush()

    return stl_loss


def process_run(
        curr_stream: torch.cuda.Stream,
        curr_run: torch.Tensor,
        curr_interval: List,
        # curr_goal: torch.Tensor,
        pred_trajs: torch.Tensor,
        threshold: List,
        sims: List,
        annots: List,
        action_mask: torch.Tensor,
        device: torch.device,
) -> Tuple[Tuple, stlcg.Expression]:
    """
    Processes each stream for the current run on a GPU kernel.

    Args:
        curr_stream (`torch.cuda.Stream`):
            An instance of the torch.cuda.Stream class.
        curr_run (`torch.Tensor`):
            The current run's waypoints to process.
        curr_interval (`List`):
            The current run's interval to process.
        curr_goal (`torch.Tensor`):
            The current run's goal to process.
        pred_trajs (`torch.Tensor`):
            The policy's predicted trajectories to perform comparison with.
        threshold (`List`):
            Threshold values for STL similarity expressions.
        sims (`List`):
            List of similarity metrics to visualize.
        annots (`List`):
            List of annotations to visualize.

    Returns:
        `Tuple`:
            The current inputs and STL expression for the current run.
    """
    def reduce_similarity(similarity: torch.Tensor):
        """
        Reduces actions to 1D value.
        """
        while similarity.dim() > 1:
            similarity = similarity.mean(dim=-1)

        return similarity.mean()
        # return (similarity * action_mask).mean() / (action_mask.mean() + 1e-2)

    with torch.cuda.stream(curr_stream):
        inner_exp = None
        inner_inputs = ()

        for j in range(len(curr_run)):
            curr_wp = curr_run[j].to(device)

            # map each waypoint to the batch of predicted trajectories
            subgoal_sim = reduce_similarity(F.cosine_similarity(
                curr_wp.unsqueeze(1).expand(-1, pred_trajs.size(0), -1, -1),
                pred_trajs.unsqueeze(0).expand(curr_wp.size(0), -1, -1, -1),
                dim=-1,
            )).unsqueeze(0).unsqueeze(0).unsqueeze(0)

            # for visualization
            sims.append(to_numpy(subgoal_sim).flatten())
            annots.append(f"w_{len(sims)}")

            # recursively add subgoal metric to inner inputs
            if len(inner_inputs) == 0:
                inner_inputs = subgoal_sim
            else:
                inner_inputs = (inner_inputs, subgoal_sim)

            # access the current interval
            interv = curr_interval[j]

            # add subgoal to STL formula
            subgoal_sim = stlcg.Eventually(
                stlcg.Expression("phi_ij", subgoal_sim) > threshold[0],
                interval=[interv[0], interv[1]]
            )

            # recursively add subgoal exp to inner exp
            if inner_exp is None:
                inner_exp = subgoal_sim
            else:
                inner_exp &= subgoal_sim  # test AND vs OR

    return inner_inputs, inner_exp
