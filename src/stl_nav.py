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
from typing import Tuple, Union
import os
import warnings
import time
import gc
import argparse
import heapq
import shutil

from viz_utils import *
from text_utils import *

# meant for MobileViT arch, be aware of real warnings
warnings.filterwarnings("ignore")

from mobile_sam import sam_model_registry, SamAutomaticMaskGenerator
from vit_pytorch.mobile_vit import MobileViT
from diffusers import StableDiffusionPipeline
# from transformers import CLIPTokenizer
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
- negative prompts/STL examples

- check wandb list saved
- see if obs vision encoder tokens can be inputted
- explore SDF w/ text-to-depth images

- clean up code (comment, delete unnecessary snippets, ...) 
- get sacson to work again
- remove process_data + vint_train dirs

COMPLETED
- testing intervals that reset every batch...
- object persistence
- added no grads to inference, sorted out memory issues
"""

login(os.getenv("HUGGINGFACE_HUB_TOKEN"))
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:21000"


def load_models(device: torch.device): 
    """
    Loads the pre-trained ViT models onto the specified device.

    Args:
        device: if GPU not available, use CPU.
    """
    weights_dir = os.path.join(os.getcwd(), "pretrained_weights")

    # sam_dir = "pretrained_weights/weight/sam_vit_h_4b8939.pth"
    deeplab_dir = os.path.join(
            weights_dir,
            "deeplab/best_deeplabv3plus_resnet101_cityscapes_os16.pth.tar"
    )

    mbvit_config = {
        'image_size': (256, 256),
        'dims': [96, 120, 144],
        'channels': [16, 32, 48, 48, 64, 64, 80, 80, 96, 96, 384],
        #'channels': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], 
        'num_classes': 1000,
    }

    # semantic segmentation model
    # deep_lab = torch.hub.load("pytorch/vision:v0.10.0", "deeplabv3_resnet50", pretrained=True).eval()
    deeplab = modeling.__dict__["deeplabv3plus_resnet101"](num_classes=19, output_stride=8)
    deeplab.load_state_dict(torch.load(deeplab_dir)["model_state"])

    # test: non-semantic segmentation model
    # # specify ViT mask generators 
    # weight_path = os.path.join(os.getcwd(), sam_dir)
    # model_type = 'vit_h'
    # model_class = sam_model_registry[model_type](checkpoint=weight_path).to(device=device).eval()
    # seg_model = SamAutomaticMaskGenerator(model_class)

    # image encoder
    mb_vit = MobileViT(
            image_size=mbvit_config['image_size'],
            dims=mbvit_config['dims'],
            channels=mbvit_config['channels'],
            num_classes=mbvit_config['num_classes'],
    )
    #.to(device)

    # text-to-image models
    # token_model = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
    pipe = StableDiffusionPipeline.from_pretrained(
            "CompVis/stable-diffusion-v1-4",
            cache_dir=os.path.join(weights_dir, "stable_diffusion"),
            use_auth_token=True,
    )
    
    return deeplab.to(device), mb_vit.to(device), pipe.to(device)
    # return seg_model, mb_vit
    # return mb_vit


@torch.no_grad()
def generate_latents_from_obs(
    obs_imgs: torch.Tensor,
    models: tuple,
    gen_subgoal: bool,
    device: torch.device,
    data_pos: int = 0,
    visualize: bool = False,
) -> Union[Tuple, torch.Tensor]:
    """
    Generates latents from the observation needed for STL.

    Args:
        obs_imgs: observation images shaped 96x96.
        models: semantic segmentation, encoder, and text-to-image generators.
        device: the device to move tensors onto during inference.
        gen_subgoal: specifies pre-training or training process.
        data_pos: position in the dataset loop for calculating intervals.
        visualize: visualize obs segmentation images with WandB.

    Returns:
        `tuple` or torch.Tensor:
            If gen_subgoal is true, z_batch with intervals will be returned,
            otherwise a `tuple` is returned with no interval array.
    """
    # load in all models
    seg_model, enc_model, gen_model = models

    enc_img_size = (256, 256)  # for MobileViT inputs

    if torch.cuda.is_available():
        obs_imgs = obs_imgs.to(device)

    if gen_subgoal:
        # with torch.no_grad():
        outputs = seg_model(obs_imgs)
        preds, intervals = filter_preds(outputs, data_pos)

        print(f"PREDS SHAPE {len(preds)}, INTERVALS SHAPE {intervals.shape}")

        # ====================== MODEL OUTPUTS ===============
        # input_ids = gen_model.tokenizer(preds,
        #         padding=True,
        #         return_tensors="pt"
        # ).input_ids.to(device)

        # text_embeddings = gen_model.text_encoder(input_ids).last_hidden_state
        # size = gen_model.unet.sample_size
        # latents = torch.randn((text_embeddings.shape[0], gen_model.unet.in_channels, size, size)).to(device)
        # print(text_embeddings.shape)
        # print(latents.shape)
        # num_steps = 50
        # for t in range(num_steps):
        #     timesteps = torch.tensor([num_steps - t - 1], dtype=torch.long).to(device)
        #     # timesteps = timesteps.expand(228, -1)
        #
        #     with autocast("cuda"):
        #         print(gen_model.unet.sample_size)
        #         noise_pred = gen_model.unet(latents, timesteps, encoder_hidden_states=text_embeddings).sample
        #
        #     latents -= noise_pred * gen_model.scheduler.sigmas[t]

        subgoal_imgs = []
        dis_imgs = []
        step_size = 3
        # limit = 10
        for i in range(0, len(preds), step_size):
            # if i == limit:
            #     break
            output_imgs = gen_model(preds[i: i + 3]).images
            output_imgs = [TF.pil_to_tensor(img) for img in output_imgs]
            output_imgs = [TF.resize(img, enc_img_size) for img in output_imgs]

            # print(f"shape of output_imgs: {len(output_imgs)}")

            fig, ax = plt.subplots(step_size)
            save_path = f"subgoal_{i}.png"
            for j in range(len(output_imgs)):
                ax[j].imshow(np.moveaxis(to_numpy(output_imgs[j]), 0, -1))
            plt.savefig(save_path)
            dis_imgs.append(wandb.Image(save_path))

            # output_img = TF.pil_to_tensor(output_imgs[0])
            # output_img = TF.resize(output_img, enc_img_size)
            subgoal_imgs.extend(output_imgs)
        # intervals = intervals[:limit]

        # nothing else is needed except the images and intervals
        del preds
        gc.collect()
        torch.cuda.empty_cache()

        print("LOGGED!")
        wandb.log({"ex": dis_imgs}, commit=False)
        # raise Exception("test")

        subgoal_imgs = torch.stack(subgoal_imgs).float().to(device)
        print(subgoal_imgs.shape)
        # with torch.no_grad():
        z_batch = (enc_model(subgoal_imgs), intervals)

        del subgoal_imgs
        del intervals
        gc.collect()
        torch.cuda.empty_cache()

        # if visualize:
        #     visualize_segmentation(obs_imgs, preds)  # return an extra un-normalized pred to visualize

        return z_batch
    else:
        gc.collect()
        torch.cuda.empty_cache()

        obs_imgs = TF.resize(obs_imgs, enc_img_size)
        # no gradient accumulation works wonders
        # with torch.no_grad():
        z_t = enc_model(obs_imgs)

        del obs_imgs
        gc.collect()
        torch.cuda.empty_cache()

        return z_t

    # # generate masks
    # obs_imgs = np.moveaxis(to_numpy(obs_imgs), 1, -1)
    # obs_imgs = (obs_imgs * 255.).astype("uint8")

    # masks = seg_model.generate(obs_img)

    # # preprocess mask images to be encoded
    # mask_img = mask_to_img(masks)
   
    # # encode mask image
    # # z_t = mb_vit(mask_img)

    # print("shape before passing to vit", obs_imgs.shape)

    # gc.collect()
    # torch.cuda.empty_cache()

    # the whole point is to process the images in parallel
    # z_t = mb_vit(obs_imgs.to(device))
    # z_t = mb_vit(obs_imgs)


def generate_waypoints(
    dataset: DataLoader,
    models: tuple,
    device: torch.device,
    load_waypoints: bool = True,
    visualize: bool = False,
) -> torch.Tensor:
    """
    Generates latent waypoints from the full, unshuffled dataset.

    Args:
        dataset: unshuffled dataset to load batches in.
        seg_model: semantic segmentation model for class labels.
        enc_model: encoder model for waypoint latents.
        gen_model: text-to-image model for subgoal generation.
        device: current memory device.
        load_waypoints: determines whether to generate or load waypoint files.
        visualize: waypoint visualization boolean.

    Returns:
        z_w: the full waypoint latents as a tensor.
    """
    z_stacked = None

    # load last latent file if not already saved
    latent_dir = "latents"
    latent_file = os.path.join(latent_dir, "last_saved_diffusion.pt")

    # ensure directory is created
    if not os.path.isdir(latent_dir):
        os.makedirs(latent_dir)

    # # TODO: TEMP
    # shutil.rmtree(latent_dir)

    wandb_images = []
    batch_size = 256  # for intervals

    # TODO: test
    #gc.collect()
    #torch.cuda.empty_cache()

    preprocess = transforms.Compose([
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # if the folder has no files, generate new latents
    if not load_waypoints:  # TODO: change to automate
        for idx, data in enumerate(tqdm.tqdm(dataset, desc="Generating waypoints...")):
            start_time = time.time()

            obs_img, _, _, _, _, _, _ = data

            obs_imgs = torch.split(obs_img, 3, dim=1)

            # concatenate everything together (not needed)
            # obs_imgs = torch.cat(obs_imgs, dim=0)

            # reshape to MobileViT-compatible shape
            # obs_imgs = TF.resize(obs_imgs[-1], [256, 256])

            # reshape to DeepLab compatible shape
            obs_imgs = [preprocess(obs_img) for obs_img in obs_imgs]
            obs_imgs = torch.stack(obs_imgs)[-1]  # TODO: see if taking more images suffices

            # convert the resized batch into latent vectors
            z_batch = generate_latents_from_obs(
                    obs_imgs=obs_imgs,
                    models=models,
                    gen_subgoal=True,
                    device=device,
                    # data_pos=batch_size * idx,
                    data_pos=0,  # TODO: TEMP================
                    visualize=False,
            )
            print(f"shape of the stack: {z_batch[0].shape}, {z_batch[1].shape}")

            assert z_batch[0].shape[0] == z_batch[1].shape[0]

            # save the visualization images (optional)
            if visualize:
                for img in obs_imgs:
                    save_path = f"observation_{i}_{j}.png"

                    # change dims for plotting
                    obs_img = np.moveaxis(to_numpy(img), 0, -1)
                    plt.imshow(obs_img)
                    plt.savefig(save_path)

                    wandb_images.append(wandb.Image(save_path))
                    wandb.log({"ex": wandb_images}, commit=False)

            torch.save(z_batch, latent_file + f"_{idx}")

            print("ELAPSED TIME: %.2f s." % (time.time() - start_time))

            # TODO: implement goal images
            # viz_goal_img = TF.resize(goal_img, VISUALIZATION_IMAGE_SIZE[::-1])

            # recursively concatenate latent vectors
            # if z_stacked is None:
            #     z_stacked = z_w
            # else:
            #     z_stacked = torch.cat((z_stacked, z_w), dim=0)

            # # TODO: temporary stop point
            # if z_batch[0].size(0) > 2000:
            #     break

            # for obs in viz_obs_img:
            #     print("Generating embeddings from observation...")
            #     z_t = generate_embeddings_from_obs(
            #             obs,
            #             seg_model,
            #             mb_vit
            #             )  # TODO: error may occur here
            #     
            #     if z_stacked is None:
            #         z_stacked = z_t
            #     else:    
            #         print("\nBEFORE")
            #         print(z_stacked.shape)
            #         print(z_t.shape)
            #         z_stacked = torch.cat((z_stacked, z_t), dim=0)
            #         print("AFTER\n", z_stacked.shape)

            #     # TODO: this is temporary
            #     if z_stacked.shape[0] == 100:
            #         break
            # break  # TODO: temp

        # torch.save(z_stacked, latent_file)
        # torch.save(z_batch, latent_file)
        print("Successfully saved waypoint latents!")

        # TODO: Program doesn't need to continue at this point
        sys.exit()
    else:
        z_w = None
        intervals = None
        for root, dirs, files in os.walk(latent_dir):
            for f in tqdm.tqdm(files, "Loading waypoints..."):
                batch = torch.load(os.path.join(root, f))
                
                if z_w is None:
                    z_w = batch[0]
                    intervals = batch[1]
                else:
                    z_w = torch.cat((z_w, batch[0]))
                    intervals = np.concatenate((intervals, batch[1]))

        del batch  # final batch should be freed
        gc.collect()

        assert z_w.shape[0] == intervals.shape[0]

        print(f"Successfully loaded waypoint latents! Shapes: {z_w.shape}, {intervals.shape}")

    return z_w, intervals


def compute_stl_loss(
    obs_imgs: torch.Tensor, 
    wp_latents: torch.Tensor, 
    device: torch.device,
    # formula: stlcg.STL_Formula,
    models: tuple,
    visualize: bool = False,
) -> torch.Tensor:
    """
    Generates STL formula and inputs for robustness, while computing the STL
    loss based on robustness.
    """
    start_time = time.time()

    # don't need intervals for observation latents
    obs_latents = generate_latents_from_obs(
            obs_imgs=obs_imgs,
            models=models,
            device=device,
            gen_subgoal=False,
    )

    intervals = wp_latents[1]  # tricky, careful about ordering here
    wp_latents = wp_latents[0].reshape(-1, wp_latents[0].size(-1)).to(device)

    print(f"time: {time.time() - start_time}")

    # for each waypoint, compute cos_sim for the observation batch
    outer_form = None

    # indicates the sensitivity of satisfaction for cosine distance
    THRESHOLD = 0.5

    # signal inputs to the robustness function
    inputs = ()

    # Method 1: O(nm) time

    # print(wp_latents.shape, obs_latents.shape)

    # ensure that all latent observations are close to at least one waypoint latent
    # for i in range(wp_latents.shape[0]):
    #     inner_form = None
    #     for j in range(obs_latents.shape[0]):
    #         # print(i, j)
    #         # print(wp_latents[i].shape)
    #         # print(obs_latents[j].shape)
    #         cossim = F.cosine_similarity(wp_latents[i], obs_latents[j], dim=0).float()
    #         inputs.append(cossim)
    #         cossim = stlcg.Expression("phi_ij", cossim) > THRESHOLD

    #         if inner_form is None:
    #             inner_form = cossim
    #         else:
    #             inner_form = inner_form & cossim
    #     if outer_form is None:
    #         outer_form = inner_form
    #     else:
    #         outer_form = outer_form | inner_form

    # Method 2: O(n) with Until temporal operator

    # inputs = []
    # formula = None
    # for z_t in z_embeddings:
    #     cossim = F.cosine_similarity(z_t, z, dim=1).float()
    #     inputs.append(cossim)

    #     # cossim = 2 * torch.randn(1) - 1

    #     # atomic proposition for cosine similarity threshold
    #     phi_t = stlcg.Expression('phi_t', cossim) > 0  # TODO: consider threshold value as well

    #     if formula is None:
    #         formula = phi_t
    #     else:
    #         formula = stlcg.Until(formula, phi_t, interval=[0,1])  # overlap=False TODO: check interval

    # Method 3: O(m), process observation costs in parallel

    # outer_form = None
    # THRESHOLD = 0.6  # TODO: tweak this, learnable parameter?

    # for i in range(wp_latents.shape[0]):
    #     # process observation latents in parallel
    #     cossim = F.cosine_similarity(
    #             wp_latents[i].unsqueeze(0),
    #             obs_latents,
    #             dim=1
    #     ).unsqueeze(-1).unsqueeze(-1)

    #     # base cases and recursive concatenation
    #     if len(inputs) == 0:
    #         inputs = (cossim,)
    #     elif len(inputs) == 1:
    #         inputs = (inputs[0], cossim)
    #     else:
    #         inputs = (inputs, cossim)

    #     # inputs.append(cossim.unsqueeze(-1).unsqueeze(-1))

    #     cossim = stlcg.Expression("phi_j", cossim) > THRESHOLD

    #     if outer_form is None:
    #         outer_form = cossim
    #     else:
    #         outer_form = outer_form | cossim

    #TODO: temp inputs
    # print(inputs[0].shape, inputs[1].shape, inputs[2].shape)
    # inputs = ((inputs[0], inputs[1]), inputs[2])
    # print(inputs)
    # inputs = torch.tensor(inputs)
    # print("inputs==========", min(inputs[0]), max(inputs[0]))
    # print("cossim", F.cosine_similarity(wp_latents[0].unsqueeze(0),
        # torch.rand(wp_latents[0].shape)))

    # Method 4: O(1): Method 3 + temporal relationship
    
    """
    Broadcast each latent waypoint to all latent obs:
    (1) Broadcast wp_latents to element-wise mult with obs_latents.
    (2) Broadcast obs_latents to each wp_latent.
    """
    cos_sims = F.cosine_similarity(
           wp_latents[:, None, :].expand(-1, obs_latents.size(0), -1),
           obs_latents[None, :, :].expand(wp_latents.size(0), -1, -1),
           dim=-1
    )

    print(cos_sims)

    print("cossim shape:", cos_sims.shape)

    assert cos_sims.shape[0] == intervals.shape[0]

    for i, cos_sim in enumerate(cos_sims):
        # signals must have at least 3 dims
        cos_sim = cos_sim[None, None, :]

        # generate inputs w/ recursive concatenation
        if len(inputs) == 0:
            inputs = (cos_sim,)
        elif len(inputs) == 1:
            inputs = (inputs[0], cos_sim)
        else:
            inputs = (inputs, cos_sim)

        # TODO: formula created at training time, can change to reduce complexity
        # overloading notation, cos_sim is now an STL expression
        # print("curr interval", intervals[i])
        cos_sim = stlcg.Eventually(
                stlcg.Expression("phi_j", cos_sim) > THRESHOLD,
                interval=[intervals[i][0], intervals[i][1]]
        )

        if outer_form is None:
            outer_form = cos_sim
        else:
            outer_form = outer_form | cos_sim
     
    # print(outer_form)

    if outer_form is None:
        raise ValueError(f"Formula is not properly defined.")
    
    # saves a digraph of the STL formula
    if visualize:
        digraph = viz.make_stl_graph(outer_form)
        viz.save_graph(digraph, "utils/formula")
        print("Saved formula CG successfully.")

    # compute the robustness
    margin = 0.0
    # print(inputs)
    robustness = (-outer_form.robustness(inputs)).squeeze() 
    stl_loss = F.leaky_relu(robustness - margin).mean()

    print(f"STL loss: {stl_loss}, \tTime (s): {time.time() - start_time}")
    
    return stl_loss
