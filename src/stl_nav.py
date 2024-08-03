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
- policy in STL

- reduce STL time?
- make sure each batch is 1 run
- parallelize wp generation process
- lerp, slerp, etc...
- try leakyrelu
- negative prompts/STL examples
- faster text-to-image model
- automate params on cmd line

- our diffusion model is parallelized across multiple GPUs using nn.DataParallel
"""

login(os.getenv("HUGGINGFACE_HUB_TOKEN"))
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:21000"
sys.setrecursionlimit(10000)  # needed for STL robustness

# TODO: watch out for these lines
# reset img and latent directories
if not os.path.isdir(IMG_DIR):
    os.makedirs(IMG_DIR, exist_ok=True)
else:
    if RESET_IMG_DIR:
        shutil.rmtree(IMG_DIR)
        os.makedirs(IMG_DIR, exist_ok=True)

if not os.path.isdir(LATENT_DIR):
    os.makedirs(LATENT_DIR, exist_ok=True)
# else:
#     if RESET_LATENT_DIR:
#         shutil.rmtree(LATENT_DIR)
#         os.makedirs(LATENT_DIR, exist_ok=True)

subgoal_dir = os.path.join(LATENT_DIR, "subgoal")
goal_dir = os.path.join(LATENT_DIR, "goal")

if not os.path.isdir(subgoal_dir):
    os.makedirs(subgoal_dir, exist_ok=True)

if not os.path.isdir(goal_dir):
    os.makedirs(goal_dir, exist_ok=True)


def load_models(device: torch.device) -> Tuple:
    """
    Loads the pre-trained ViT models onto the specified device.

    Args:
        device (torch.device): if GPU is not available, use CPU.

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


@torch.no_grad()
def generate_latents_from_obs(
        mode: str,
        obs_imgs: torch.Tensor,
        models: tuple,
        device: torch.device,
        enable_intervals: bool,
        data_pos: int = 0,
        visualize_subgoals: bool = VISUALIZE_SUBGOALS,
) -> Union[Tuple, torch.Tensor]:
    """
    Generates latents from the observation, depending on the mode argument provided.

    Args:
        mode (str): determines the modes for encoding latents.
        obs_imgs (torch.Tensor): observation images shaped 96x96.
        models (tuple): semantic segmentation, encoder, and text-to-image generators.
        gen_subgoal (bool): specifies pre-training or training process.
        device (torch.device): the device to move tensors onto during inference.
        enable_intervals (bool): enable the generation of intervals for each image.
        data_pos (int): position in the dataset loop for calculating intervals.
        visualize_subgoals (bool): save subgoal images to corresponding `IMG_DIR` directory.

    Returns:
        `tuple` or torch.Tensor:
            If enable_intervals is false, z_batch without intervals will be returned,
            otherwise a `tuple` is returned with an intervals array.
    """
    # load in all models
    seg_model, enc_model, gen_model = models
    enc_img_size = (256, 256)  # for MobileViT inputs

    if torch.cuda.is_available():
        obs_imgs = obs_imgs.to(device)

    # for MobileViT inputs
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((256, 256)),
    ])

    if mode == "pretraining":
        with torch.no_grad():
            outputs = seg_model(obs_imgs)

        preds, intervals = filter_preds(outputs, data_pos)

        # we cannot continue process if there are no valid waypoints
        if len(preds) == 0:
            print("Found no predictions, skipping...")
            return

        subgoal_imgs = []

        step_size = 3
        for i in range(0, len(preds), step_size):
            with torch.no_grad():
                output_imgs = gen_model(prompt=preds[i: i + 3]).images

            output_imgs = [preprocess(img) for img in output_imgs]
            output_imgs = torch.stack(output_imgs)

            subgoal_imgs.extend(output_imgs)

        if visualize_subgoals:
            num_imgs = len(subgoal_imgs)
            rows, cols = (int(np.sqrt(num_imgs)), int(np.sqrt(num_imgs)))

            fig, ax = plt.subplots(rows, cols)
            save_path = os.path.join(IMG_DIR, f"subgoal_{num_imgs}.png")

            for i in range(rows):
                for j in range(cols):
                    ax[i, j].imshow(np.moveaxis(to_numpy(subgoal_imgs[i * rows + j]), 0, -1))
                    ax[i, j].axis("off")

            plt.tight_layout(rect=[0, 0, 1, 0.9])
            plt.subplots_adjust(wspace=0.1, hspace=0.1)
            fig.suptitle(f"{rows * cols} subgoal imgs, Δt = {PERSISTENCE_THRESH}", fontsize=10, y=0.95)
            plt.savefig(save_path, bbox_inches="tight")
            plt.close(fig)
            print("PLOTTED")

        subgoal_imgs = torch.stack(subgoal_imgs).to(device)
        print("SUBGOAL IMGS", subgoal_imgs.shape)

        with torch.no_grad():
            z_batch = enc_model(subgoal_imgs).last_hidden_state
            # z_batch is contiguous
            z_batch = z_batch.view(z_batch.size(0), -1)
            if enable_intervals:
                z_batch = (z_batch, intervals)

        print("ENCODED LATENTS", z_batch[0].shape)  # only if there is an interval

        # if visualize:
        #     visualize_segmentation(obs_imgs, preds)  # return an extra un-normalized pred to visualize

        return z_batch
    else:
        flush()

        # resize to valid MobileViT input size
        obs_imgs = TF.resize(obs_imgs, enc_img_size)

        # no gradient accumulation significantly reduces memory footprint
        with torch.no_grad():
            z_t = enc_model(obs_imgs, output_hidden_states=True).last_hidden_state
            # print(z_t.shape)  # (256, 640, 8, 8)
            # print(enc_model(obs_imgs).hidden_states)
            # print(enc_model(obs_imgs).hidden_states.shape)
            z_t = z_t.view(z_t.size(0), -1)
            # print(z_t.shape)  # (1, 40960)

        # if processing the goals, interpolate latents with equal weights
        if mode == "goal":
            n = z_t.size(0)
            # transpose weights matrix and compute weighted sum
            weights = (torch.ones(n) / n).view(1, -1).to(device)
            z_t = weights @ z_t

            # check that interpolation is done properly
            assert z_t.size(0) == 1

        # print(f"ENCODED OBS: {z_t.shape}, DEVICE: {z_t.device}")
        del obs_imgs
        flush()

        return z_t


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
        dataset (Dataloader): unshuffled dataset to load batches in.
        models (tuple): all model for segmentation, text-to-image, and encoding.
        device (torch.device): encoder model for waypoint latents.
        enable_intervals (bool): enable interval creation and loading.
        load_waypoints (bool): determines whether to generate or load waypoint files.
        visualize_dataset (bool): determines the visualization of the dataset images.
        visualize_subgoals (bool): determines the visualization of generated subgoal images.
        process_subgoals (bool): determines subgoal processing.
        process_goals (bool): determines goal processing.

    Returns:
        Value of type `None` or `tuple`:
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

            # when plotting, each image is different
            if visualize_dataset:
                batch_viz_obs_images = TF.resize(obs_imgs[-1], VISUALIZATION_IMAGE_SIZE[::-1])
                batch_viz_goal_images = TF.resize(goal_img, VISUALIZATION_IMAGE_SIZE[::-1])

                # num_plots = 3
                batch_len = len(batch_viz_obs_images)

                # don't put figs right next to each other
                rows = int(np.sqrt(batch_len))
                cols = int(np.sqrt(batch_len))

                fig, ax = plt.subplots(rows, cols)

                save_path = os.path.join(IMG_DIR, f"observation_{idx}.png")

                obs_imgs = to_numpy(batch_viz_obs_images)
                obs_imgs = np.moveaxis(obs_imgs, 1, -1)

                # change idx based on batch or single img
                for r in range(rows):
                    for c in range(cols):
                        ax[r, c].imshow(obs_imgs[r * rows + c])
                        ax[r, c].axis('off')  # Hide axis

                plt.tight_layout()
                plt.subplots_adjust(wspace=0.1, hspace=0.1)
                plt.savefig(save_path)
                plt.close(fig)

                fig2, ax2 = plt.subplots(rows, cols)

                save_path = os.path.join(IMG_DIR, f"goal_{idx}.png")

                goal_imgs = to_numpy(batch_viz_goal_images)
                goal_imgs = np.moveaxis(goal_imgs, 1, -1)

                # change idx based on batch or single img
                for r in range(rows):
                    for c in range(cols):
                        ax2[r, c].imshow(goal_imgs[r * rows + c])
                        ax2[r, c].axis('off')  # Hide axis

                plt.tight_layout()
                plt.subplots_adjust(wspace=0.1, hspace=0.1)
                plt.savefig(save_path)
                plt.close(fig2)

                # TODO: temp
                sys.exit()

                # wandb_images.append(wandb.Image(save_path))
                # wandb.log({"ex": wandb_images}, commit=False)

            # process images to latent vectors
            if process_subgoals:
                # obs are resized in terms of how MobileViT dimensions are calculated
                obs_imgs = [preprocess(img) for img in obs_imgs]
                obs_imgs = torch.stack(obs_imgs)[-1]  # TODO: see if taking more images suffices

                # process obs latents
                z_batch = generate_latents_from_obs(
                    mode="pretraining",
                    obs_imgs=obs_imgs,
                    models=models,
                    device=device,
                    data_pos=0,
                    visualize_subgoals=visualize_subgoals,
                    enable_intervals=enable_intervals,
                )

                print(f"shape of returned z batch: {len(z_batch[0])}, {len(z_batch[1])}")

                if z_batch is not None:
                    torch.save(z_batch, subgoal_latent_file)
                    print("Successfully saved waypoint latents!")

                    if enable_intervals:
                        assert len(z_batch[0]) == len(z_batch[1])

                del obs_imgs, z_batch
                flush()

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
        else:
            z_w = None
            for root, dirs, files in os.walk(latent_dir):
                for f in tqdm.tqdm(files, "Loading waypoints..."):
                    batch = torch.load(os.path.join(root, f))

                    if z_w is None:
                        z_w = batch[0]
                    else:
                        z_w = torch.cat((z_w, batch[0]))

            print(f"Successfully loaded waypoint latents! Shapes: {z_w.shape}")

            return z_w


def compute_stl_loss(
        # obs_imgs: torch.Tensor,
        obs_latents: torch.Tensor,
        batch_latents: torch.Tensor,
        goal_latents: torch.Tensor,
        device: torch.device,
        models: tuple,
        streams: List,
        margin: int = 0.05,
        enable_intervals: bool = ENABLE_INTERVALS,
        visualize_stl: bool = VISUALIZE_STL,
        threshold: List = SIM_THRESH,  # indicates the sensitivity of satisfaction for similarity distance
) -> torch.Tensor:
    """
    Generates STL formula and inputs for robustness, while computing the STL loss based on robustness.

    Broadcast each target latent to all latent obs during similarity computation.
    (1) Broadcast target latent to element-wise multiply with obs latents.
    (2) Broadcast obs latent to each target.

    Args:
        obs_latents (torch.Tensor): online training observations.
        batch_latents (torch.Tensor): waypoint latents to perform similarity with + intervals.
        goal_latents (torch.Tensor): goal latents to perform similarity with.
        device (torch.device): device to transfer tensors onto.
        models (tuple): encoder needed for observation latents.
        streams (List): a list of torch.cuda.Stream's instances to deploy runs on a multi-stream setup.
        margin (int): margin in the ReLU objective.
        enable_intervals (bool): boolean determining the inclusion of intervals in encoding process.
        visualize_stl (bool): boolean determining the STL formula visualization.
        threshold (list): threshold values for STL similarity expressions.
    Returns:
         stl_loss (torch.Tensor): the computed STL loss with no loss weight.
    """
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

    # transform latents to match obs condition vector for similarity computation
    wp_latents = [run.reshape(-1, obs_latents.size(0)) for run in wp_latents]
    print(f"WP LATENT SHAPE: {np.array(wp_latents).shape}")
    goal_latents = [goal.reshape(-1, obs_latents.size(0)) for goal in goal_latents]
    print(f"GOAL LATENT SHAPE: {np.array(wp_latents).shape}")

    # full expression used for training
    full_exp = None
    # signal inputs to the robustness function
    full_inputs = ()
    # used for similarity visualization
    sims = []
    annots = []

    test_start = time.time()

    # multi-stream setup
    # streams = [torch.cuda.Stream() for _ in range(2)]

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
        elif len(full_inputs) == 1:
            full_inputs = (*full_inputs, inner_inputs)
        else:
            full_inputs = (full_inputs, inner_inputs)

        # print(full_inputs)
        # print(len(full_inputs))

        # recursively add inner exp to full exp
        if full_exp is None:
            full_exp = inner_exp
        else:
            full_exp |= inner_exp

    for stream in streams:
        stream.synchronize()

    if VISUALIZE_SIM:
        fig, ax = plt.subplots()
        x = range(0, len(sims) * 100, 100)
        ax.plot(x[:-1], sims[:-1], marker='o', linestyle='-', label='Similarities')
        ax.plot(x[-1], sims[-1], marker='o', linestyle='-', color="red", label='Similarities')

        for i, a in enumerate(annots):
            ax.annotate(a, (x[i], sims[i]))

        ax.set_xlabel("Time")
        ax.set_ylabel("Similarity Metrics")
        plt.title(f"{len(wp_latents)} Runs, 1 Training Iteration")
        plt.savefig(os.path.join(IMG_DIR, f"run_{len(sims)}.png"))

        print(f"COSSIM RECURSIVE Time (s): {time.time() - test_start}")

    if full_exp is None:
        raise ValueError(f"Formula is not properly defined.")

    # print(full_exp)
    # print(full_inputs)
    # print(len(tuple(full_exp.__str__())))
    # print(len(full_inputs))

    # saves a digraph of the STL formula
    if visualize_stl:
        digraph = viz.make_stl_graph(full_exp)
        viz.save_graph(digraph, "utils/formula")
        print("Saved formula CG successfully.")

    # recursive depth may result in overloading stack memory
    robustness = (-full_exp.robustness(full_inputs)).squeeze()
    stl_loss = torch.relu(robustness - margin).mean()

    print(f"STL loss: {stl_loss}, \tSTL Computation Time (s): {time.time() - start_time}")

    return stl_loss

    # for i in range(len(wp_latents)):
    #     curr_latent = wp_latents[i]
    #     inner_exp = None
    #     inner_inputs = ()
    #
    #     loop_start = time.time()
    #
    #     print("before curr latent", i, f"len curr latent: {len(curr_latent)}")
    #
    #     for j in range(len(curr_latent)):
    #         inner_start = time.time()
    #
    #         # compute subgoal similarity and convert to valid STL input
    #         subgoal_sim = F.cosine_similarity(
    #             curr_latent[j].unsqueeze(0).expand(obs_latents.size(0), -1),
    #             obs_latents,
    #             dim=-1
    #         ).mean().unsqueeze(0).unsqueeze(0).unsqueeze(0)
    #
    #         # for visualization
    #         sims.append(to_numpy(subgoal_sim).flatten())
    #         annots.append(f"w_{len(sims)}")
    #
    #         # recursively add subgoal metric to inner inputs
    #         if len(inner_inputs) <= 1:
    #             inner_inputs = (*inner_inputs, subgoal_sim)
    #         else:
    #             inner_inputs = (inner_inputs, subgoal_sim)
    #
    #         print(f"time after subgoal: {time.time() - inner_start}")
    #
    #         # access the current interval
    #         curr_interval = intervals[i][j]
    #
    #         # add subgoal to STL formula
    #         subgoal_sim = stlcg.Eventually(
    #             stlcg.Expression("phi_ij", subgoal_sim) > threshold[0],
    #             interval=[curr_interval[0], curr_interval[1]]
    #         )
    #
    #         # recursively add subgoal exp to inner exp
    #         if inner_exp is None:
    #             inner_exp = subgoal_sim
    #         else:
    #             inner_exp |= subgoal_sim  # TODO: test AND
    #
    #     full_start = time.time()
    #
    #     # compute goal similarity and convert to valid STL input
    #     goal_sim = F.cosine_similarity(
    #         goal_latents[i].expand(obs_latents.size(0), -1),  # TODO: see if this works
    #         obs_latents,
    #         dim=-1
    #     ).mean().unsqueeze(0).unsqueeze(0).unsqueeze(0)
    #
    #     print("finished goal sim")
    #
    #     # visualization, must move tensor to CPU
    #     sims.append(to_numpy(goal_sim).flatten())
    #     annots.append("g")
    #
    #     if len(sims) % 500 == 0:
    #         fig, ax = plt.subplots()
    #         x = range(len(sims))
    #         ax.plot(x[:-1], sims[:-1], marker='o', linestyle='-', label='Similarities')
    #         ax.plot(x[-1], sims[-1], marker='o', linestyle='-', color="red", label='Similarities')
    #
    #         for i, a in enumerate(annots):
    #             ax.annotate(a, (x[i], sims[i]))
    #
    #         ax.set_xlabel("Time")
    #         ax.set_ylabel("Similarity Metrics")
    #         plt.title(f"Multiple Run")
    #         plt.savefig(os.path.join(IMG_DIR, f"run_{len(sims)}.png"))
    #
    #     print(f"time after goal: {time.time() - full_start}")
    #
    #     # add goal metric to inner inputs
    #     inner_inputs = (inner_inputs, goal_sim)
    #
    #     # recursively add inner inputs to full inputs
    #     if len(full_inputs) <= 1:
    #         full_inputs = (*full_inputs, inner_inputs)
    #     else:
    #         full_inputs = (full_inputs, inner_inputs)
    #
    #     # add goal to STL formula
    #     inner_exp = stlcg.Until(inner_exp, stlcg.Expression("varphi_i", goal_sim) > threshold[1])
    #
    #     # recursively add inner exp to full exp
    #     if full_exp is None:
    #         full_exp = inner_exp
    #     else:
    #         full_exp |= inner_exp
    #
    #     print(f"loop time: {time.time() - loop_start}")
    #
    # # print(f"LEN EXP: {len(full_exp)}, LEN INPUTS: {len(full_inputs)}")
    # print(f"COSSIM RECURSIVE Time (s): {time.time() - test_start}")
    #
    # if full_exp is None:
    #     raise ValueError(f"Formula is not properly defined.")
    #
    # print(full_exp)
    # print(full_inputs)
    #
    # # saves a digraph of the STL formula
    # if visualize_stl:
    #     digraph = viz.make_stl_graph(full_exp)
    #     viz.save_graph(digraph, "utils/formula")
    #     print("Saved formula CG successfully.")
    #
    # # recursive depth may result in overloading stack memory
    # robustness = (-full_exp.robustness(full_inputs)).squeeze()
    # stl_loss = torch.relu(robustness - margin).mean()
    #
    # print(f"STL loss: {stl_loss}, \tTime (s): {time.time() - start_time}")
    #
    # return stl_loss


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
        curr_stream (torch.cuda.Stream): an instance of the torch.cuda.Stream class.
        curr_run (torch.Tensor): the current run's waypoints to process.
        curr_interval (List): the current run's interval to process.
        curr_goal (torch.Tensor): the current run's goal to process.
        obs_latents (torch.Tensor): all observation latents to perform comparison with.
        threshold (List): threshold values for STL similarity expressions.
        sims (List): list of similarity metrics to visualize.
        annots (List): list of annotations to visualize.

    Returns:
        inner_inputs, inner_exp (Tuple): the current inputs and STL expression for the current run.
    """
    with torch.cuda.stream(curr_stream):
        inner_exp = None
        inner_inputs = ()

        # loop_start = time.time()

        for j in range(len(curr_run)):
            # inner_start = time.time()

            curr_wp = curr_run[j]

            # compute subgoal similarity and convert to valid STL input
            subgoal_sim = F.cosine_similarity(
                curr_wp.unsqueeze(0).expand(obs_latents.size(0), -1),
                obs_latents,
                dim=-1
            ).mean().unsqueeze(0).unsqueeze(0).unsqueeze(0)

            # for visualization
            sims.append(to_numpy(subgoal_sim).flatten())
            annots.append(f"w_{len(sims)}")

            # recursively add subgoal metric to inner inputs
            if len(inner_inputs) <= 1:
                inner_inputs = (*inner_inputs, subgoal_sim)
            else:
                inner_inputs = (inner_inputs, subgoal_sim)

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
                inner_exp &= subgoal_sim  # test AND

        # full_start = time.time()

        # compute goal similarity and convert to valid STL input
        goal_sim = F.cosine_similarity(
            curr_goal.expand(obs_latents.size(0), -1),
            obs_latents,
            dim=-1
        ).mean().unsqueeze(0).unsqueeze(0).unsqueeze(0)

        # print("finished goal sim")

        # visualization, must move tensor to CPU
        sims.append(to_numpy(goal_sim).flatten())
        annots.append("g")

        # print(f"time after goal: {time.time() - full_start}")

        if len(inner_inputs) <= 1:
            inner_inputs = (*inner_inputs, goal_sim)
        else:
            inner_inputs = (inner_inputs, goal_sim)

        # add goal to STL formula
        goal_sim = stlcg.Expression("varphi_i", goal_sim) > threshold[1]
        inner_exp = stlcg.Until(inner_exp, goal_sim)

        # print(f"loop time: {time.time() - loop_start}")

    return inner_inputs, inner_exp
