import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torchvision import transforms
from torch.utils.data import DataLoader
from torch import autocast

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
import cv2
import tqdm
from PIL import Image
import imageio
import wandb
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

import sys
from typing import Tuple, Union, List
import os
import warnings
import time
import gc
import shutil

from viz_utils import *
from text_utils import *
from constants import *

sys.path.insert(0, "stlcg/src")  # disambiguate path names
import stlcg
import stlviz as viz

# loading in deeplabv3 custom repo
sys.path.insert(0,  "DeepLabV3Plus-Pytorch")
from network import modeling


os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:21000"
# STL recursion depth may overload the stack memory
sys.setrecursionlimit(10000)

# initiate paths
traj_path = os.path.join(WP_DIR, TRAJ_DIR)
subgoal_dir = os.path.join(traj_path, "subgoal")
goal_dir = os.path.join(traj_path, "goal")

if not os.path.isdir(subgoal_dir):
    os.makedirs(subgoal_dir, exist_ok=True)

if not os.path.isdir(goal_dir):
    os.makedirs(goal_dir, exist_ok=True)

if not os.path.isdir(IMG_DIR):
    os.makedirs(IMG_DIR, exist_ok=True)
# else:
#     if RESET_IMG_DIR:
#         shutil.rmtree(IMG_DIR)
#         os.makedirs(IMG_DIR, exist_ok=True)
#
frame_dir = os.path.join(IMG_DIR, "frames")
if not os.path.isdir(frame_dir):
    os.makedirs(frame_dir, exist_ok=True)

imgs = []
for root, dir, files in os.walk(frame_dir):
    for file in files:
        imgs.append(file)

imgs = sorted(imgs, key=lambda x: int(x[x.find("_", x.find("_") + 1) + 1: x.find(".")]))

# load images using OpenCV
images = []
for img_path in imgs:
    full_path = os.path.join(frame_dir, img_path)
    img = cv2.imread(full_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
    images.append(img)

# save images as a GIF
with imageio.get_writer(os.path.join(IMG_DIR, 'obs_0.gif'), mode='I', duration=1000/3, loop=0) as writer:
    for image in images[:900]:  # Select first 900 images
        writer.append_data(image)
sys.exit()
# # PIL doesn't support plt alpha values
# # imgs = [Image.open(os.path.join(frame_dir, img_path)) for img_path in imgs]
# # imgs = [img.convert("RGBA") for img in imgs]
#
# # imgs[0].save(os.path.join(IMG_DIR, "obs_0.gif"), save_all=True, append_images=imgs[1:900], duration=1000/7, loop=0)



def reduce_similarity(similarity: torch.Tensor) -> torch.Tensor:
    """
    Performs mean reduction along dimensions to a resulting vector.

    Args:
        similarity (`torch.Tensor`): metric or value to reduce.

    Returns:
        `torch.Tensor`:
            The reduced value.
    """
    while similarity.dim() > 1:
        similarity = similarity.mean(dim=-1)

    # return similarity.mean()
    return similarity
    # return (similarity * action_mask).mean() / (action_mask.mean() + 1e-2)


def cossim_method(
        inner_inputs: Tuple,
        inner_exp: None,
        curr_run: List,
) -> Tuple[Tuple, stlcg.Expression]:
    """
    Cosine similarity method. Constructs the STL formula based on the angle of predicted trajectory and waypoint
    vectors.

    Args:
        inner_inputs (`tuple`):
        inner_exp (`None`):
        curr_run (`torch.Tensor`):

    Returns:
        `tuple`:
            The inner inputs used for the overall robust ness function, and the inner boolean expression for
            the full STL expression.
    """

    for j in range(len(curr_run)):
        curr_wp = curr_run[j].to(device)

        # map each waypoint to the batch of predicted trajectories
        subgoal_sim = reduce_similarity(F.cosine_similarity(
            curr_wp.unsqueeze(1).expand(-1, pred_trajs.size(0), -1, -1),
            pred_trajs.unsqueeze(0).expand(curr_wp.size(0), -1, -1, -1),
            dim=-1,
        )).unsqueeze(0).unsqueeze(0).unsqueeze(0)

        # for visualization
        sims.append(subgoal_sim.detach().cpu().numpy().flatten())
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

    # # construct STL formula based on everything afterward
    # for i in range(len(wp_sims)):
    #     subgoal_sim = wp_sims[i].unsqueeze(0).unsqueeze(0).unsqueeze(0)
    #
    #     # for visualization
    #     sims.append(subgoal_sim.detach().cpu().numpy().flatten())
    #     annots.append(f"w_{len(sims)}")
    #
    #     # recursively add subgoal metric to inner inputs
    #     if len(inner_inputs) == 0:
    #         inner_inputs = subgoal_sim
    #     else:
    #         inner_inputs = (inner_inputs, subgoal_sim)
    #
    #     # add subgoal to STL formula
    #     subgoal_sim = stlcg.Expression("phi_ij", subgoal_sim) > threshold[0]
    #
    #     # recursively add subgoal exp to inner exp
    #     if inner_exp is None:
    #         inner_exp = subgoal_sim
    #     else:
    #         inner_exp = stlcg.Until(inner_exp, subgoal_sim)

    return inner_inputs, inner_exp


def mse_method(
        curr_run: List,
        pred_trajs: torch.Tensor,
        threshold: List,
        goal_pos: torch.Tensor,
        inner_inputs: Tuple,
        inner_exp: None,
        device: torch.device,
        visualize_traj: bool,
        dataset_idx: int,
        gamma: float = 1.12,
) -> Tuple[Tuple, stlcg.Expression]:
    """
    MSE loss method. Constructs the STL formula s.t. it satisfies all predicted observation trajectories
    at once. Each trajectory prediction has similarity metrics with all waypoint paths.

    Args:
        curr_run (`List`):
            List of waypoints for the current run of observations.
        pred_trajs (`torch.Tensor`):
            Predicted trajectories for the current batch of observations.
        threshold (`List`):
            Similarity threshold for STL waypoint expression.
        goal_pos (`torch.Tensor`):
            Ground truth goal position drawn from the dataset.
        inner_inputs (`tuple`):
            Inner recursive inputs to the robustness function for the current run.
        inner_exp (`stlcg.Expression`):
            Inner expression for the current run.
        device (`torch.device`):
            The device to move tensors onto.
        visualize_traj (`bool`):
            Whether to visualize waypoint and predicted trajectories.
        gamma (`float`, defaults to 1.12):
            Growth factor for threshold.
        dataset_idx (`int`):
            Current iteration of the training loop.

    Returns:
        `tuple`:
            The inner inputs used for the overall robustness function, and the inner boolean expression for
            the full STL expression.
    """

    # move trajs to GPU
    curr_run = [wp.to(device) for wp in curr_run]
    pred_trajs.to(device)

    # initialize list to store similarity metrics
    traj_sims = []

    for curr_wp in curr_run:
        # map each predicted trajectory to the wps
        traj_sim = reduce_similarity(F.mse_loss(
            curr_wp.unsqueeze(0).expand(pred_trajs.size(0), -1, -1, -1),
            pred_trajs.unsqueeze(1).expand(-1, curr_wp.size(0), -1, -1),
            reduction="none",
        ))

        # each pred trajectory should have a similarity metric with the wp's paths
        assert len(traj_sim) == 256, "Metric is not computed correctly"

        traj_sims.append(traj_sim)

    # each observation trajectory should have a similarity metric with each waypoint
    traj_sims = torch.stack(traj_sims).transpose(0, 1)

    # print(f"num wps: {len(curr_run)}")  # varies
    # print(f"traj_sims shape {traj_sims.shape}")  # (256, num_waypoints)
    assert len(traj_sims) == 256, "traj_sims is not converted to tensor properly"

    # retrieve the closest waypoint to each pred traj
    closest_indices = torch.argmax(traj_sims, dim=1)
    # print(f"closest_indices shape {closest_indices.shape}")  # (256,)
    assert len(closest_indices) == 256, "There are less than the acceptable amount of observations"

    start_time = time.time()

    # constructs STL formula based on w_{i} -> w_{i + 1} -> ..., where i = argmax(traj_sims)

    # temp = []

    for i in range(len(traj_sims)):
        # the closest wp in the current obs predicted traj
        closest_wp_idx = closest_indices[i].item()
        # filtered similarity metrics for the current pred traj
        filtered_sims = traj_sims[i][closest_wp_idx:]

        # temp.append(filtered_sims.mean())

        inner_inner_inputs = ()
        inner_inner_exp = None

        # gradually increase the threshold for future similarities
        discounted_thresholds = [gamma ** j * threshold[0] for j in range(len(filtered_sims))]

        # constructs a seq of filtered wp similarities: w_{i} U w_{i+1} U w_{i+2} ...
        for j in range(len(filtered_sims)):
            # current wp in the filtered sequence
            wp_sim = filtered_sims[j].unsqueeze(0).unsqueeze(0).unsqueeze(0)

            if len(inner_inner_inputs) == 0:
                inner_inner_inputs = wp_sim
            else:
                inner_inner_inputs = (inner_inner_inputs, wp_sim)

            # similarity expression for the current waypoint
            wp_sim = stlcg.Expression(f"w_{i}_{closest_wp_idx}", wp_sim) < discounted_thresholds[j]

            if inner_inner_exp is None:
                inner_inner_exp = wp_sim
            else:
                inner_inner_exp = stlcg.Until(inner_inner_exp, wp_sim)

        if len(inner_inputs) == 0:
            inner_inputs = inner_inner_inputs
        else:
            inner_inputs = (inner_inputs, inner_inner_inputs)

        # Constructs (w_{i} U w_{i+1} U w_{i+2} ...) ⋀ (w_{j} U w_{j+1} U w_{j+2} ...) ⋀ ...
        if inner_exp is None:
            inner_exp = inner_inner_exp
        else:
            inner_exp &= inner_inner_exp

    # print(f"full mean {torch.stack(temp).mean()}")

    if visualize_traj:
        # plot the filtered wp traj with the corresponding pred traj

        for i in range(len(closest_indices)):
            # if i % 50 != 0:
            #     continue

            fig, ax = plt.subplots()
            x_limit = 25
            y_limit = 25

            # w_{argmax_{t} pred_traj}, w_{t+1}, ...
            wp_sequence = [wp.detach().cpu().numpy() for wp in curr_run[closest_indices[i].item():]]  # (w, t, 8, 2)
            # detach first for less computational overhead
            pred_obs_traj = pred_trajs[i].detach().cpu().numpy()[None]

            # each waypoint has shape (t, 8, 2)
            traj_list = np.concatenate([
                *wp_sequence,  # (w, t, 8, 2)
                pred_obs_traj,  # (1, 8, 2),
            ], axis=0)

            # generate RGB colors for each waypoint path
            wp_colors = list(colors.CSS4_COLORS.keys())[:len(wp_sequence)]

            # each set of waypoint trajs receives 1 color
            wp_seq_colors = [wp_colors[i] for i in range(len(wp_sequence)) for _ in wp_sequence[i]]
            traj_colors = wp_seq_colors + ["green"] * len(pred_obs_traj)
            traj_alphas = [0.7] * sum(len(wp) for wp in wp_sequence) + [1.0] * len(pred_obs_traj)

            # make points numpy array of robot positions (0, 0) and goal positions
            point_list = [np.array([0, 0]), goal_pos[i].detach().cpu().numpy()]
            point_colors = ["black", "yellow"]
            point_alphas = [1.0, 1.0]

            traj_labels = [f"w_{i}" for i in range(len(wp_sequence)) for _ in wp_sequence[i]] + ["pred"]

            # 20
            # print(f"f len pred_obs_traj: {len(pred_obs_traj)}")
            # print(f"len traj_labels: {len(traj_labels)}")
            # print(f"len traj_list: {len(traj_list)}")
            # print(f"len traj_colors: {len(traj_colors)}")
            # print(f"len traj_alphas: {len(traj_alphas)}")

            # plot the trajectories and start/end points
            plot_trajs_and_points(
                ax,
                traj_list,
                point_list,
                traj_colors,
                point_colors,
                traj_labels=traj_labels,
                point_labels=["robot", "goal"],
                quiver_freq=0,
                traj_alphas=traj_alphas,
                point_alphas=point_alphas,
                frame_dir=frame_dir,
            )

            ax.set_title("Predicted Trajectory Against Waypoints")
            ax.set_xlim(-x_limit, x_limit)
            ax.set_ylim(-y_limit, y_limit)

            save_path = os.path.join(frame_dir, f"obs_{i}_{dataset_idx}.png")
            plt.savefig(save_path)
            plt.close(fig)

            break  # TODO: only for first one for now

        # sys.exit()  # TODO: for now

    return inner_inputs, inner_exp


def process_run(
        curr_stream: torch.cuda.Stream,
        curr_run: List,
        pred_trajs: torch.Tensor,
        # curr_goal: torch.Tensor,
        goal_pos: torch.Tensor,
        threshold: List,
        curr_interval: List,
        sims: List,
        annots: List,
        action_mask: torch.Tensor,
        device: torch.device,
        visualize_traj: bool,
        dataset_idx: int,
) -> Tuple[Tuple, stlcg.Expression]:
    """
    Processes each stream for the current run on a GPU kernel. Constructs an STL formula for each run, which is to be
    concatenated into the full expression in the compute_stl_loss function.

    Args:
        curr_stream (`torch.cuda.Stream`):
            An instance of the torch.cuda.Stream class.
        curr_run (`torch.Tensor`):
            The current run's waypoints to process.
        pred_trajs (`torch.Tensor`):
            The policy's predicted trajectories to perform comparison with.
        curr_goal (`torch.Tensor`):
            The current run's goal to process.
        goal_pos (`torch.Tensor`):
            Goal position used for plotting.
        threshold (`List`):
            Threshold values for STL similarity expressions.
        curr_interval (`List`):
            The current run's interval to process.
        sims (`List`):
            List of similarity metrics to visualize.
        annots (`List`):
            List of annotations to visualize.
        action_mask (`torch.Tensor`):
            Action mask used for clipping action outputs.
        device (`torch.Tensor`):
            Device to move tensors onto.
        dataset_idx (`int`):

        visualize_traj (`bool`):

    Returns:
        `Tuple`:
            The current inputs and STL expression for the current run.
    """

    with torch.cuda.stream(curr_stream):
        inner_exp = None
        inner_inputs = ()

        return mse_method(
            curr_run=curr_run,
            pred_trajs=pred_trajs,
            threshold=threshold,
            goal_pos=goal_pos,
            inner_inputs=inner_inputs,
            inner_exp=inner_exp,
            device=device,
            visualize_traj=visualize_traj,
            dataset_idx=dataset_idx,
        )

        # return cossim_method(
        #     inner_inputs=inner_inputs,
        #     inner_exp=inner_exp,
        #     ax=ax,
        #     curr_run=curr_run,
        #     goal_pos=goal_pos,
        # )


def compute_stl_loss(
        pred_trajs: torch.Tensor,
        wp_data: torch.Tensor,
        goal_data: torch.Tensor,
        device: torch.device,
        streams: List,
        dataset_idx: int,
        action_mask: torch.Tensor,
        goal_pos: torch.Tensor,
        margin: int = 0,  # TODO: change
        visualize_stl: bool = VISUALIZE_STL,
        vis_freq: int = VIS_FREQ,
        threshold: List = SIM_THRESH,
        weight_stl: float = WEIGHT_STL,
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
        dataset_idx (`int`):
            The current dataset iteration.
        goal_pos (`torch.Tensor`):
            The goal coordinates for the current batch.
        margin (`int`, defaults to 0):
           Margin in the ReLU objective.
        visualize_stl (`bool`, defaults to `VISUALIZE_STL`):
            Determines the STL formula visualization.
        vis_freq (`int`, defaults to `VIS_FREQ`):
            Visualization frequency for plotting similarity metrics.
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
        return None, None

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

    # processes a multi-stream setup for runs on parallel GPU kernels
    for i, stream in enumerate(streams):
        # start_time = time.time()

        # only one run
        full_inputs, full_exp = process_run(
            curr_stream=stream,
            curr_run=wp_trajs[i],
            # curr_goal=goal_trajs[i],
            pred_trajs=pred_trajs,
            goal_pos=goal_pos,
            threshold=threshold,
            curr_interval=intervals[i],
            sims=sims,
            annots=annots,
            action_mask=action_mask,
            device=device,
            visualize_traj=True,
            dataset_idx=dataset_idx,
        )
        break

        # # TODO: temp, only plot the first run, of the first pred traj, of the first wp
        # if i == 0:
        #     visualize_traj = True
        # else:
        #     visualize_traj = False
        #
        # inner_inputs, inner_exp = process_run(
        #     curr_stream=stream,
        #     curr_run=wp_trajs[i],
        #     # curr_goal=goal_trajs[i],
        #     pred_trajs=pred_trajs,
        #     goal_pos=goal_pos,
        #     threshold=threshold,
        #     curr_interval=intervals[i],
        #     sims=sims,
        #     annots=annots,
        #     action_mask=action_mask,
        #     device=device,
        #     visualize_traj=visualize_traj,
        #     dataset_idx=dataset_idx,
        # )
        #
        # # full input tuples are not concerned with individual values
        # if len(full_inputs) == 0:
        #     full_inputs = inner_inputs
        # else:
        #     full_inputs = (full_inputs, inner_inputs)
        #
        # # recursively add inner exp to full exp
        # if full_exp is None:
        #     full_exp = inner_exp
        # else:
        #     full_exp |= inner_exp

        # print(f"Stream {i} time: {time.time() - start_time}")

    for stream in streams:
        stream.synchronize()

    if full_exp is None:
        raise ValueError(f"Formula is not properly defined.")

    # print(f"full_exp: {full_exp}")
    # print(f"full_inputs: {full_inputs}")

    robustness_time = time.time()
    robustness = full_exp.robustness(full_inputs).squeeze()
    print(f"elapsed time for robustness: {time.time() - robustness_time}")
    robustness_time = time.time()
    stl_loss = torch.relu(-robustness - margin).mean()
    print(f"elapsed time for stl loss: {time.time() - robustness_time}")

    # saves a digraph of the STL formula
    if visualize_stl:
        digraph = viz.make_stl_graph(full_exp)
        viz.save_graph(digraph, "utils/formula")
        print("Saved formula CG successfully.")

    if VISUALIZE_SIM:
        if dataset_idx % vis_freq == 0:
            sims = sims[:20]
            annots = annots[:20]

            fig, ax = plt.subplots()
            x = range(0, len(sims) * 100, 100)
            ax.plot(x, sims, marker='o', linestyle='-', label='Similarities')

            for i, a in enumerate(annots):
                ax.annotate(a, (x[i], sims[i]))

            ax.set_xlabel("Time")
            ax.set_ylabel("Similarity Metrics")
            plt.title(f"{len(wp_trajs)} Runs, 1 Training Iteration")
            plt.savefig(os.path.join(IMG_DIR, f"run_{dataset_idx}.png"))

        print(f"COSSIM RECURSIVE Time (s): {time.time() - test_start}")

    print(f"STL loss: {weight_stl * stl_loss}, \tSTL Compute Time (s): {time.time() - start_time}")

    flush()

    return robustness, weight_stl * stl_loss


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
        print("Finished saving waypoints!")

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

            print(f"Successfully loaded waypoints!\n"
                  f"# wp: {len(w_traj)}, # intervals: {len(intervals)}, # goals: {len(g_traj)}\n")

            assert len(w_traj) == len(intervals),\
                "The number of subgoal/goal latent files do not match."

            return (w_traj, intervals), None  # TODO goals


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
