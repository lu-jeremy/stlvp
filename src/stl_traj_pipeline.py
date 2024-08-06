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
import argparse
import heapq
import shutil

from viz_utils import *
from text_utils import *
from constants import *

# meant for MobileViT arch, be aware of real warnings
warnings.filterwarnings("ignore")

from diffusers import StableDiffusionPipeline
from transformers import MobileViTModel
from huggingface_hub import login

sys.path.insert(0, "visualnav-transformer/train")
from vint_train.data.data_utils import VISUALIZATION_IMAGE_SIZE
from vint_train.visualizing.visualize_utils import to_numpy, from_numpy

sys.path.insert(0, "stlcg/src")  # disambiguate path names
import stlcg
import stlviz as viz

# loading in deeplabv3 custom repo
sys.path.insert(0, "DeepLabV3Plus-Pytorch")
from network import modeling

"""
TODO
- test sacson
- 
"""

login(os.getenv("HUGGINGFACE_HUB_TOKEN"))
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:21000"
sys.setrecursionlimit(10000)  # needed for STL robustness


def load_models(device: torch.device) -> Tuple:
    """
    Loads the pre-trained ViT models onto the specified device.

    Args:
        device (`torch.device`):
            if GPU is not available, use CPU.

    Returns:
        `tuple` of all models used during the STL creation process.
    """
    weights_dir = os.path.join(os.getcwd(), "pretrained_weights")

    deeplab_dir = os.path.join(
        weights_dir,
        "deeplab/best_deeplabv3plus_resnet101_cityscapes_os16.pth.tar"
    )

    # semantic segmentation model
    deeplab = modeling.__dict__["deeplabv3plus_resnet101"](num_classes=19, output_stride=8).eval()
    deeplab.load_state_dict(torch.load(deeplab_dir)["model_state"])

    mb_vit = MobileViTModel.from_pretrained(
        "apple/mobilevit-small",
        cache_dir=os.path.join(weights_dir, "stable_diffusion"),
        use_auth_token=True).eval()

    # text-to-image models
    # token_model = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
    pipe = StableDiffusionPipeline.from_pretrained(
        "CompVis/stable-diffusion-v1-1",
        cache_dir=os.path.join(weights_dir, "stable_diffusion"),
        use_auth_token=True,
    )

    return deeplab.to(device), mb_vit.to(device), pipe.to(device)


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

    preprocess = transforms.Compose([
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # if the folder has no files, generate new latents
    if not load_waypoints:
        for idx, data in enumerate(tqdm.tqdm(dataset, desc="Generating waypoints...")):
            # no need to overwrite the same files, skip if detected
            subgoal_latent_file = os.path.join(subgoal_dir, f"{idx}.pt")
            goal_latent_file = os.path.join(goal_dir, f"{idx}.pt")

            if process_subgoals:
                if os.path.isfile(subgoal_latent_file):
                    print("SKIP")
                    continue

            if process_goals:
                if os.path.isfile(goal_latent_file):
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

            # process images to latent vectors
            if process_subgoals:
                # obs are resized in terms of how MobileViT dimensions are calculated
                obs_imgs = [preprocess(img) for img in obs_imgs]
                obs_imgs = torch.stack(obs_imgs)[-1].to(device)  # TODO: see if taking more images suffices

                # process obs latents
                seg_model, _, _ = models

                with torch.no_grad():
                    outputs = seg_model(obs_imgs)

                preds, intervals = filter_preds(outputs, data_pos)

                # we cannot continue process if there are no valid waypoints
                if len(preds) == 0:
                    print("Found no predictions, skipping...")
                    return



                # print(f"shape of returned z batch: {len(z_batch[0])}, {len(z_batch[1])}")

                # if z_batch is not None:
                #     torch.save(z_batch, subgoal_latent_file)
                #     print("Successfully saved waypoint latents!")
                #
                #     if enable_intervals:
                #         assert len(z_batch[0]) == len(z_batch[1])

                # del obs_imgs, z_batch
                # flush()

            if process_goals:
                # process goal latents, TODO CHECK IF THIS WORKS
                goal_imgs = [preprocess(img) for img in goal_img]
                goal_imgs = torch.stack(goal_imgs)

                z_goal = generate_latents_from_obs(
                    mode="goal",
                    obs_imgs=goal_imgs,
                    models=models,
                    device=device,
                    data_pos=0,
                    visualize_subgoals=visualize_subgoals,
                    enable_intervals=enable_intervals,
                )

                if z_goal is not None:
                    torch.save(z_goal, goal_latent_file)
                    print("Successfully saved goal latents!")

                del goal_imgs, z_goal
                flush()

            # just to be safe
            del obs_imgs, data
            flush()

            # print("END OF LOOP\n", torch.cuda.memory_summary())
            print("ELAPSED TIME: %.2f s." % (time.time() - start_time))

        # torch.save(z_batch, latent_file)
        print("Finished saving waypoint latents!")

        # TODO: Program doesn't need to continue at this point
        sys.exit()
    else:
        if enable_intervals:
            # z_w = None
            # intervals = None
            z_w = []
            intervals = []
            z_g = []

            # subgoal takes priority, as some preds are skipped during processing
            for root, dirs, files in os.walk(subgoal_dir):
                for f in tqdm.tqdm(files, "Loading waypoints..."):
                    batch = torch.load(os.path.join(root, f))

                    z_w.append(batch[0])
                    intervals.append(batch[1])

                    # add matching latent file with subgoal
                    batch = torch.load(os.path.join(goal_dir, f))
                    z_g.append(batch)

                    # if z_w is None:
                    #     z_w = batch[0]
                    #     intervals = batch[1]
                    # else:
                    #     z_w = torch.cat((z_w, batch[0]))
                    #     intervals = np.concatenate((intervals, batch[1]))

            print(f"Successfully loaded waypoint latents!\n"
                  f"# wp: {len(z_w)}, # intervals: {len(intervals)}, # goals: {len(z_g)}\n")

            assert len(z_w) == len(intervals) and len(intervals) == len(z_g),\
                "The number of subgoal/goal latent files do not match."

            # print(f"Successfully loaded waypoint latents! Shapes: {z_w.shape}, {intervals.shape}")

            return (z_w, intervals), z_g


def compute_stl_loss(
        # obs_imgs: torch.Tensor,
        obs_latents: torch.Tensor,
        batch_latents: torch.Tensor,
        goal_latents: torch.Tensor,
        device: torch.device,
        models: tuple,
        streams: List,
        dataset_idx: int,
        margin: int = 0,  # TODO: change
        enable_intervals: bool = ENABLE_INTERVALS,
        visualize_stl: bool = VISUALIZE_STL,
        viz_freq: int = 50,
        threshold: List = SIM_THRESH,  # indicates the sensitivity of satisfaction for similarity distance
) -> torch.Tensor:
    """
    Generates STL formula and inputs for robustness, while computing the STL loss based on robustness.

    Broadcast each target latent to all latent obs during similarity computation.
    (1) Broadcast target latent to element-wise multiply with obs latents.
    (2) Broadcast obs latent to each target.

    Args:
        obs_latents (`torch.Tensor`):
            Online training observations.
        batch_latents (`torch.Tensor`):
            Waypoint latents to perform similarity with + intervals.
        goal_latents (`torch.Tensor`):
            Goal latents to perform similarity with.
        device (`torch.device`):
            Device to transfer tensors onto.
        models (`tuple`):
            Encoder for observation latents.
        streams (`List`):
            List of torch.cuda.Stream's instances to deploy runs on a multi-stream setup.
        margin (`int`):
           Margin in the ReLU objective.
        enable_intervals (`bool`):
            Determines the inclusion of intervals in encoding process.
        visualize_stl (`bool`):
            Determines the STL formula visualization.
        threshold (`List`):
            Threshold values for STL similarity expressions.
    Returns:
        `torch.Tensor`:
            The computed STL loss with no loss weight.
    """
    flush()

    batch_size = 256
    if obs_latents.size(0) < batch_size:
        print(f"NOT LARGE ENOUGH: {obs_latents.size()}")
        return torch.tensor(0.0)

    start_time = time.time()

    # don't need intervals for observation latents
    # obs_latents = generate_latents_from_obs(
    #     mode="training",
    #     obs_imgs=obs_imgs,
    #     models=models,
    #     device=device,
    #     enable_intervals=enable_intervals,
    #     visualize_subgoals=False,
    # )

    # unpacking tuple
    wp_latents = batch_latents[0]
    intervals = batch_latents[1]

    # full expression used for training
    full_exp = None
    # signal inputs to the robustness function
    full_inputs = ()
    # used for similarity visualization
    sims = []
    annots = []

    test_start = time.time()

    print(f"len wp latents: {len(wp_latents)}, len intervals: {len(intervals)}, len obs latents: {len(obs_latents)}")

    for i, stream in enumerate(streams):
        inner_inputs, inner_exp = process_run(
            curr_stream=stream,
            curr_run=wp_latents[i],
            curr_interval=intervals[i],
            curr_goal=goal_latents[i],
            obs_latents=obs_latents,
            threshold=threshold,
            sims=sims,
            annots=annots,
        )

        # print(inner_inputs)
        # print(full_inputs)
        #
        # print(len(inner_inputs))
        # print(len(full_inputs))

        # recursively add inner inputs to full inputs
        if len(full_inputs) == 0:
            full_inputs = inner_inputs
        # elif len(full_inputs) == 1:
        #     full_inputs = (*full_inputs, inner_inputs)
        # # if len(full_inputs) <= 1:
        # #     full_inputs = (*full_inputs, inner_inputs)
        else:
            full_inputs = (full_inputs, inner_inputs)

        # print(full_inputs)
        # print(len(full_inputs))

        # recursively add inner exp to full exp
        if full_exp is None:
            full_exp = inner_exp
        else:
            full_exp |= inner_exp

    # print(full_exp)
    # print(full_inputs)

    for stream in streams:
        stream.synchronize()

    if VISUALIZE_SIM:
        if dataset_idx % viz_freq == 0:
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
            plt.title(f"{len(wp_latents)} Runs, 1 Training Iteration")
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
        curr_goal: torch.Tensor,
        obs_latents: torch.Tensor,
        threshold: List,
        sims: List,
        annots: List,
) -> Tuple[Tuple, stlcg.Expression]:
    """
    Processes each stream for the current run.

    Args:
        curr_stream (`torch.cuda.Stream`):
            An instance of the torch.cuda.Stream class.
        curr_run (`torch.Tensor`):
            The current run's waypoints to process.
        curr_interval (`List`):
            The current run's interval to process.
        curr_goal (`torch.Tensor`):
            The current run's goal to process.
        obs_latents (`torch.Tensor`):
            All observation latents to perform comparison with.
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
    with torch.cuda.stream(curr_stream):
        inner_exp = None
        inner_inputs = ()

        # loop_start = time.time()

        for j in range(len(curr_run)):
            # inner_start = time.time()

            # transform latents to match obs condition vector for similarity computation
            curr_wp = curr_run[j].reshape(-1, obs_latents.size(0))  # (160, 256)
            # simple computation purposes, linear interpolation
            # weights = (torch.ones(len(curr_wp), device=curr_wp.device) / len(curr_wp)).view(1, -1)
            # curr_wp = weights @ curr_wp

            # compute subgoal similarity and convert to valid STL input
            subgoal_sim = F.cosine_similarity(
                # curr_wp.unsqueeze(0).expand(obs_latents.size(0), -1),
                # curr_wp.expand(obs_latents.size(0), -1),
                # obs_latents,
                curr_wp.unsqueeze(1).expand(-1, obs_latents.size(0), -1),
                obs_latents.unsqueeze(0).expand(curr_wp.size(0), -1, -1),
                dim=-1
            ).mean().unsqueeze(0).unsqueeze(0).unsqueeze(0)

            # print("SUBGOAL", subgoal_sim)

            # print(inner_inputs)

            # for visualization
            sims.append(to_numpy(subgoal_sim).flatten())
            annots.append(f"w_{len(sims)}")

            # recursively add subgoal metric to inner inputs
            if len(inner_inputs) == 0:
                inner_inputs = subgoal_sim
            # elif len(inner_inputs) == 1:
            #     inner_inputs = (*inner_inputs, subgoal_sim)
            else:
                inner_inputs = (inner_inputs, subgoal_sim)

            # print(inner_inputs)

            # sys.exit()

            # print(f"time after subgoal: {time.time() - inner_start}")

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
