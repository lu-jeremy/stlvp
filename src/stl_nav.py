import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torchvision import transforms
from torch.utils.data import DataLoader
from torch import autocast

import sys
from typing import Tuple
import os
import warnings
import time
import gc
import argparse
from collections import namedtuple
import heapq
import shutil

import numpy as np
import matplotlib.pyplot as plt
import cv2
import tqdm
from PIL import Image
import wandb

# meant for MobileViT arch, be aware of real warnings
warnings.filterwarnings("ignore")

from mobile_sam import sam_model_registry, SamAutomaticMaskGenerator
from vit_pytorch.mobile_vit import MobileViT
from diffusers import StableDiffusionPipeline
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
- handle multiple prompts with stable diffusion
- return intervals for each waypoint, if it sees ground ignore it
- sort out memory issues
- clean up code (comment, delete unnecessary snippets, ...) 
- get sacson to work again
- remove process_data + vint_train dirs

COMPLETED
- added no grads to inference
- added deeplabv3
- added stable diffusion
"""


# print(os.getenv("HUGGINGFACE_HUB_TOKEN"))
login(os.getenv("HUGGINGFACE_HUB_TOKEN"))

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:21000"

# for visualization class color mappings
CityscapesClass = namedtuple('CityscapesClass', ['name', 'id', 'train_id', 'category', 'category_id',
                                                     'has_instances', 'ignore_in_eval', 'color'])
classes = [
    CityscapesClass('unlabeled',            0, 255, 'void', 0, False, True, (0, 0, 0)),
    CityscapesClass('ego vehicle',          1, 255, 'void', 0, False, True, (0, 0, 0)),
    CityscapesClass('rectification border', 2, 255, 'void', 0, False, True, (0, 0, 0)),
    CityscapesClass('out of roi',           3, 255, 'void', 0, False, True, (0, 0, 0)),
    CityscapesClass('static',               4, 255, 'void', 0, False, True, (0, 0, 0)),
    CityscapesClass('dynamic',              5, 255, 'void', 0, False, True, (111, 74, 0)),
    CityscapesClass('ground',               6, 255, 'void', 0, False, True, (81, 0, 81)),
    CityscapesClass('road',                 7, 0, 'flat', 1, False, False, (128, 64, 128)),
    CityscapesClass('sidewalk',             8, 1, 'flat', 1, False, False, (244, 35, 232)),
    CityscapesClass('parking',              9, 255, 'flat', 1, False, True, (250, 170, 160)),
    CityscapesClass('rail track',           10, 255, 'flat', 1, False, True, (230, 150, 140)),
    CityscapesClass('building',             11, 2, 'construction', 2, False, False, (70, 70, 70)),
    CityscapesClass('wall',                 12, 3, 'construction', 2, False, False, (102, 102, 156)),
    CityscapesClass('fence',                13, 4, 'construction', 2, False, False, (190, 153, 153)),
    CityscapesClass('guard rail',           14, 255, 'construction', 2, False, True, (180, 165, 180)),
    CityscapesClass('bridge',               15, 255, 'construction', 2, False, True, (150, 100, 100)),
    CityscapesClass('tunnel',               16, 255, 'construction', 2, False, True, (150, 120, 90)),
    CityscapesClass('pole',                 17, 5, 'object', 3, False, False, (153, 153, 153)),
    CityscapesClass('polegroup',            18, 255, 'object', 3, False, True, (153, 153, 153)),
    CityscapesClass('traffic light',        19, 6, 'object', 3, False, False, (250, 170, 30)),
    CityscapesClass('traffic sign',         20, 7, 'object', 3, False, False, (220, 220, 0)),
    CityscapesClass('vegetation',           21, 8, 'nature', 4, False, False, (107, 142, 35)),
    CityscapesClass('terrain',              22, 9, 'nature', 4, False, False, (152, 251, 152)),
    CityscapesClass('sky',                  23, 10, 'sky', 5, False, False, (70, 130, 180)),
    CityscapesClass('person',               24, 11, 'human', 6, True, False, (220, 20, 60)),
    CityscapesClass('rider',                25, 12, 'human', 6, True, False, (255, 0, 0)),
    CityscapesClass('car',                  26, 13, 'vehicle', 7, True, False, (0, 0, 142)),
    CityscapesClass('truck',                27, 14, 'vehicle', 7, True, False, (0, 0, 70)),
    CityscapesClass('bus',                  28, 15, 'vehicle', 7, True, False, (0, 60, 100)),
    CityscapesClass('caravan',              29, 255, 'vehicle', 7, True, True, (0, 0, 90)),
    CityscapesClass('trailer',              30, 255, 'vehicle', 7, True, True, (0, 0, 110)),
    CityscapesClass('train',                31, 16, 'vehicle', 7, True, False, (0, 80, 100)),
    CityscapesClass('motorcycle',           32, 17, 'vehicle', 7, True, False, (0, 0, 230)),
    CityscapesClass('bicycle',              33, 18, 'vehicle', 7, True, False, (119, 11, 32)),
    CityscapesClass('license plate',        -1, 255, 'vehicle', 7, False, True, (0, 0, 142)),
]

train_id_to_color = [c.color for c in classes if (c.train_id != -1 and c.train_id != 255)]
train_id_to_color.append([0, 0, 0])
train_id_to_color = np.array(train_id_to_color)

train_id_to_name = [c.name for c in classes if (c.train_id != -1 and c.train_id != 255)]
train_id_to_name = np.array(train_id_to_name)


def load_models(device: torch.device): 
    """
    Loads the pre-trained ViT models onto the specified device.
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

    # text-to-image model
    pipe = StableDiffusionPipeline.from_pretrained(
            "CompVis/stable-diffusion-v1-4",
            cache_dir=os.path.join(weights_dir, "stable_diffusion"),
            use_auth_token=True,
    )

    return deeplab.to(device), mb_vit.to(device), pipe.to(device)
    # return seg_model, mb_vit
    # return mb_vit


def generate_masks(viz_obs: torch.Tensor, model) -> dict:
    viz_obs = np.moveaxis(to_numpy(viz_obs), 0, -1)
    viz_obs = (viz_obs * 255.).astype("uint8")
    return model.generate(viz_obs)


def mask_to_img(masks: dict) -> np.ndarray:
    """
    Converts a binary mask to a randomized-color RGB image.
    """
    
    sorted_anns = sorted(masks, key=(lambda x: x['area']), reverse=True)
    mask_img = np.ones((masks[0]['segmentation'].shape[0], masks[0]['segmentation'].shape[1], 3))
    for ann in sorted_anns:
        m = ann['segmentation']
        mask_img[m] = np.random.random(3)
   
    mask_img = cv2.resize(mask_img, (256, 256))  # TODO: for ViT, it will be distorted
    mask_img = np.resize(mask_img, (mask_img.shape[-1], mask_img.shape[0], mask_img.shape[1]))

    mask_img = torch.from_numpy(mask_img).unsqueeze(0)
    mask_img = mask_img.to(torch.float32)  # type incompatibility otherwise

    #print("mask img shape ===", mask_img.shape)

    return mask_img  # Mobile-ViT only takes scalar doubles


def generate_latents_from_obs(
    obs_imgs: torch.Tensor,
    seg_model,
    enc_model: MobileViT,
    gen_model,
    gen_subgoal: bool,
    device: torch.device,
    data_pos: int = 0,  # TODO: fix this
    visualize: bool = False,
) -> torch.Tensor:
    """
    Generates latents from the observation needed for STL.

    Args:
        obs_imgs: observation images shaped 96x96.
        seg_model: semantic segmentation model, DeepLabV3.
        enc_model: encoder model for latent embeddings, MobileViT.
        gen_model: text-to-image model, StableDiffusion.
        device: the device to move tensors onto during inference.
        gen_subgoal: differentiates training/pre-training processes.
        data_pos: position in the dataset loop for calculating intervals.
        visualize: visualize obs segmentation images with WandB.

    Returns:
        z_batch: waypoint latents + STL intervals for the current batch.
    """

    def landmark_criteria(preds_unique: np.ndarray, n: int = 7) -> bool:
        """
        Images must have the correct object landmarks, have enough pixel
        predictions, and have enough landmarks.

        TODO: counts, object persistence

        for now, the number is equal to 7. one one hand, we want to create
        descriptive prompts for StableDiffusion, but on the other, we want to
        find one major landmark.
        we have static landmarks for now.
        relative frequency shouldn't work. counts make more sense.
        """
        # object_landmarks = [2, 3, 4, 5, 6, 7, 11, 12, 13, 14, 15, 16, 17, 18]
        # object_landmarks = np.array([2, 3, 4, 5, 6, 7])
        object_landmarks = np.array([2, 6, 7, 18])
       
        in_landmarks = np.intersect1d(preds_unique, object_landmarks, assume_unique=True)

        # skip if none of the conditions satisfy
        #return len(preds_unique) <= n and \
               #len(in_landmarks) != 0
        return len(in_landmarks) != 0

    enc_img_size = (256, 256)  # for MobileViT inputs
    intervals = []
    n = 3  # for filter criteria

    if torch.cuda.is_available():
        obs_imgs = obs_imgs.to(device)

    if gen_subgoal:
        with torch.no_grad():
            outputs = seg_model(obs_imgs)

        # return the max probability for each image
        preds = outputs.max(1)[1].detach().cpu().numpy()
        preds[preds == 255] = 19
        # format of image doesn't matter, flatten
        preds = preds.reshape(preds.shape[0], -1)

        assert(preds.shape == (256, 96 ** 2))

        preds_comb = [np.unique(p, return_counts=True)
                for p in preds]
        # preds_comb = [(set(p), count) for (p, count) in preds_comb]

        # filter out predictions
        preds = []
        # object_landmarks = set([i for i in range(19)]) - {0, 1, 8, 9, 10}
        
        # retrieve n-top unique classes
        for i, (preds_unique, counts) in enumerate(preds_comb):
            # if there aren't enough landmarks or not an object landmark, skip
            if not landmark_criteria(preds_unique, n):
                continue
            
            # don't do unnecessary computation if not necessary
            # if preds_unique.shape[0] < n:
            top_idx = np.arange(len(preds_unique))
            #else: 
            # top_idx = np.argpartition(preds_unique, -n)[-n:]
            # top_idx = heapq.nlargest(n, preds_unique)
            n_top = preds_unique[top_idx]

            preds.append(n_top)
            intervals.append([data_pos + i, data_pos + i + 1])

        del preds_comb
        gc.collect()

        intervals = np.array(intervals)
        print("INTERVAL SHAPE", intervals.shape)

        # print(preds_flat.shape)

        # preds_combined = [np.unique(p, return_counts=True) for p in preds_flat]

        # preds_unique = preds_combined[:, 0]
        # counts = preds_combined[:, 1]
        
        # print(counts)

        # freq_idx = np.argmax(counts)
        # pred_id = preds_unique[freq_idx]

        # convert to class labels based on indices
        preds = [train_id_to_name[p] for p in preds]
        print(len(preds))
        
        preds = [" and ".join(preds[i].reshape(-1)) \
                for i in range(len(preds))
        ]

        print("PROMPTS", preds[0:4])

        subgoal_imgs = []
        dis_imgs = []
        #height = 128
        #width = 128

        with torch.no_grad():
            with autocast("cuda"):
                # TODO: TEMP add full prompts, filter more
                limit = 10
                for i in range(len(preds)):
                    if i == limit:
                        break
                    output_img = gen_model(preds[i]).images[0]
                    output_img = TF.pil_to_tensor(output_img)
                    # output_img = TF.resize(output_img, enc_img_size)
                    subgoal_imgs.append(output_img)

                    save_path = f"subgoal_{i}.png"
                    plt.imshow(np.moveaxis(to_numpy(output_img), 0, -1))
                    plt.savefig(save_path)
                    dis_imgs.append(wandb.Image(save_path))
                  
                print("LOGGED!")
                wandb.log({"ex": dis_imgs}, commit=False)
                raise Exception("test")

                intervals = intervals[:limit]
                    
        # nothing else is needed except the images and intervals
        del preds
        gc.collect()
        torch.cuda.empty_cache()

        subgoal_imgs = torch.stack(subgoal_imgs).float().to(device)
        print(subgoal_imgs.shape)
        with torch.no_grad():
            z_batch = (enc_model(subgoal_imgs), intervals)

        del subgoal_imgs
        gc.collect()
        torch.cuda.empty_cache()

        if visualize:
            dis_imgs = []
            for i in range(len(preds)):
                colorized_preds = train_id_to_color[preds].astype("uint8")
                # print(colorized_preds.shape)  # (256, 256, 256, 3), which is expected
                colorized_preds = Image.fromarray(colorized_preds[i])

                # # visualize output mask classes
                # palette = torch.tensor([2 ** 25 - 1, 2 ** 15 - 1, 2 ** 21 - 1])
                # colors = torch.as_tensor([i for i in range(21)])[:, None] * palette
                # colors = (colors % 255).numpy().astype("uint8")

                # r = Image.fromarray(output_predictions.byte().cpu().numpy()).resize((160, 120))
                # r.putpalette(colors)

                # plt.imshow(r)

                mask_dir = f"mask_img_{i}.png"
                plt.imshow(colorized_preds)
                plt.savefig(mask_dir)
                dis_imgs.append(wandb.Image(mask_dir))

                obs_dir = f"obs_img_{i}.png"
                plt.imshow(np.moveaxis(obs_imgs[i].cpu().numpy(), 0, -1))
                plt.savefig(obs_dir)
                dis_imgs.append(wandb.Image(obs_dir))
                
                # requires an error or complete successful run
                if i == 9:
                    break
            wandb.log({"ex2": dis_imgs}, commit=False)

        return z_batch
    else:
        gc.collect()
        torch.cuda.empty_cache()

        obs_imgs = TF.resize(obs_imgs, enc_img_size)
        # no gradient accumulation works wonders
        with torch.no_grad():
            z_batch = (enc_model(obs_imgs), intervals)

        del obs_imgs
        gc.collect()
        torch.cuda.empty_cache()

        return z_batch 
      

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

    return z_batch


def generate_waypoints(
    dataset: DataLoader, 
    seg_model,
    enc_model: MobileViT,
    gen_model,
    device: torch.device,
    load_waypoints: bool = False,
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
            # TODO: try utilizing GPU memory
            #gc.collect()
            #torch.cuda.empty_cache()

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
                    seg_model=seg_model,
                    enc_model=enc_model,
                    gen_model=gen_model,
                    gen_subgoal=True,
                    device=device,
                    data_pos=batch_size * idx,
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

            # TODO: implement goal images
            # viz_goal_img = TF.resize(goal_img, VISUALIZATION_IMAGE_SIZE[::-1])

            # recursively concatenate latent vectors
            # if z_stacked is None:
            #     z_stacked = z_w
            # else:
            #     z_stacked = torch.cat((z_stacked, z_w), dim=0)

            torch.save(z_batch, latent_file + f"_{idx}")
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
    seg_model,
    enc_model: MobileViT,
    gen_model=None,
    visualize: bool = False,
) -> torch.Tensor:
    """
    Generates STL formula and inputs for robustness, while computing the STL
    loss based on robustness.
    """
    start_time = time.time()

    # don't need intervals for observation latents
    obs_latents, _ = generate_latents_from_obs(
            obs_imgs=obs_imgs,
            seg_model=seg_model,
            enc_model=enc_model,
            gen_model=gen_model,
            device=device,
            data_pos=None,
            gen_subgoal=False,
    )

    intervals = wp_latents[1]  # tricky, careful about ordering here
    wp_latents = wp_latents[0].reshape(-1, wp_latents[0].size(-1)).to(device)

    print(f"time: {time.time() - start_time}")

    # for each waypoint, compute cos_sim for the observation batch
    outer_form = None

    # indicates the sensitivity of satisfaction for cosine distance
    THRESHOLD = 0.9

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

    print("cossim shape:", cos_sims.shape)

    assert cos_sims.shape[0] == intervals.shape[0]

    for i, cos_sim in enumerate(cos_sims):
        # signals must have at least 3 dims
        cos_sim = cos_sim[None, None, :]

        # generate inputs w/ base cases and recursive concatenation
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


def visualize_sam_maps(
    viz_obs: torch.Tensor,
    viz_goal: torch.Tensor,
    obs_map: dict,
    goal_map: dict,
    viz_freq: int = 10,
):
    def show_anns(anns, ax):
        if len(anns) == 0:
            return

        sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
        ax.set_autoscale_on(True)

        img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
        img[:,:,3] = 0
        for ann in sorted_anns:
            m = ann['segmentation']
            color_mask = np.concatenate([np.random.random(3), [0.35]])
            img[m] = color_mask
        ax.imshow(img)

    wandb_list = []

    for i, (obs, goal) in enumerate(zip(batch_viz_obs_images, batch_viz_goal_images)):
        if i % viz_freq == 0:
            fig, ax = plt.subplots(1, 4)

            obs_img = np.moveaxis(to_numpy(obs), 0, -1)
            goal_img = np.moveaxis(to_numpy(goal), 0, -1)

            obs_img = (obs_img * 255.).astype('uint8')
            goal_img = (goal_img * 255.).astype('uint8')

            ax[0].imshow(obs_img)
            ax[1].imshow(goal_img)
            
            #save_path = os.path.join(visualize_path, 'original.png')
            #plt.savefig(save_path)
            #wandb_list.append(wandb.Image(save_path))
            #wandb.log({'examples': wandb_list}, commit=False)

            # generate masks
            # obs_map = mask_generator.generate(obs_img)
            # goal_map = mask_generator.generate(goal_img)

            #ax[2].imshow(obs_map['segmentation'])
            #ax[3].imshow(goal_map['segmentation'])

            #show_anns(obs_map, ax[0])
            #show_anns(goal_map, ax[1])
            
            show_anns(obs_map, ax[2])
            show_anns(goal_map, ax[3])

            ax[0].set_title('Observation')
            ax[1].set_title('Goal')
            ax[2].set_title('Obs Map')
            ax[3].set_title('Goal Map')

            for a in ax.flatten():
                a.axis('off')

            visualize_path = "examples"
            map_save_path = os.path.join(visualize_path, f'maps_{i}.png')
            plt.savefig(map_save_path)

            wandb_list.append(wandb.Image(map_save_path))

            wandb.log({'examples': wandb_list}, commit=False)

            print(f"Finished generating masks for maps_{i}.png.")


# IOU conversion 
#map_ious = [obs["predicted_iou"] for obs in obs_map]
#avg_iou = sum(map_ious) / len(map_ious)

#o = stlcg.Expression('obs', obs)
#g = stlcg.Expression('g', goal)

#intersection = (obs & goal).float().sum((1, 2)) + 1e-6
#union = (obs | goal).float().sum(i(1, 2)) + 1e-6
#iou = intersection / union
#iou = avg_iou * 2. - 1.  # without normalization set to >.5 


"""class STLViz:
    def __init__(
        self,
        unshuffled_dataset: torch.utils.data.DataLoader
    ):
        self.formula = self.generate_waypoints(unshuffled_dataset)
    
        # specify ViT mask generators 
        weight_path = os.path.join(os.getcwd(), "pre_trained_weights/weights/weight/sam_vit_h_4b8939.pth")
        model_type = 'vit_h'
        mobile_sam = sam_model_registry[model_type](checkpoint=weight_path).to(device=device).eval()
        self.mask_generator = SamAutomaticMaskGenerator(mobile_sam)

        # specify ViT image encoder
        mbvit_xs = MobileViT(
                image_size=mbvit_config['image_size'],
                dims=mbvit_config['dims'],
                channel=mbvit_config['channel'],
                num_classes=mbvit_config['num_classes'],
        )

    def generate_masks(obs_img: torch.Tensor):
        return self.mask_generator.generate(obs_img)

    def generate_waypoints(self, dataset) -> stlcg.STL_Formula:
        for i, data in enumerate(tqdm.tqdm(dataset, desc="Generating waypoints...")):
            obs_img, goal_img, _, _, _, _, _ = data

            obs_imgs = torch.split(obs_img, 3, dim=1)
            viz_obs_img = TF.resize(obs_imgs[-1], VISUALIZATION_IMAGE_SIZE[::-1])

            #map_ious = [obs["predicted_iou"] for obs in obs_map]
            #avg_iou = sum(map_ious) / len(map_ious)

            #o = stlcg.Expression('obs', obs)
            #g = stlcg.Expression('g', goal)

            #intersection = (obs & goal).float().sum((1, 2)) + 1e-6
            #union = (obs | goal).float().sum(i(1, 2)) + 1e-6
            #iou = intersection / union
            #iou = avg_iou * 2. - 1.  # without normalization set to >.5

            #return stlcg.Always(stlcg.Expression('iou', iou))



    def compute_stl_loss(self, viz_obs: torch.Tensor) -> dict, torch.Tensor:
        stl_loss = 0

        for i, obs in enumerate(viz_obs):
            obs = np.moveaxis(obs, 0, -1)
            obs = (obs * 255.).astype("uint8")
            obs_map = self.mask_generator(obs)

            margin = 0.05

            # STL loss
            if self.formula is None:
                raise ValueError(f"Formula is not properly defined: {self.formula}")

            robustness = (-self.formula.robustness(obs_map)).squeeze() 
            stl_loss += F.leaky_relu(robustness - margin).mean()
        
        return stl_loss / viz_obs.shape[0]  # TODO: check if this is right

    @staticmethod
    def visualize_sam_maps(
        viz_obs: torch.Tensor,
        viz_goal: torch.Tensor,
        obs_map: dict,
        goal_map: dict,
        viz_freq: int = 10,
    ):

        def show_anns(anns, ax):
            if len(anns) == 0:
                return

            sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
            ax.set_autoscale_on(True)

            img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
            img[:,:,3] = 0
            for ann in sorted_anns:
                m = ann['segmentation']
                color_mask = np.concatenate([np.random.random(3), [0.35]])
                img[m] = color_mask
            ax.imshow(img)

        wandb_list = []

        for i, (obs, goal) in enumerate(zip(batch_viz_obs_images, batch_viz_goal_images)):
            if i % viz_freq == 0:
                fig, ax = plt.subplots(1, 4)

                obs_img = np.moveaxis(to_numpy(obs), 0, -1)
                goal_img = np.moveaxis(to_numpy(goal), 0, -1)

                obs_img = (obs_img * 255.).astype('uint8')
                goal_img = (goal_img * 255.).astype('uint8')

                ax[0].imshow(obs_img)
                ax[1].imshow(goal_img)
                
                #save_path = os.path.join(visualize_path, 'original.png')
                #plt.savefig(save_path)
                #wandb_list.append(wandb.Image(save_path))
                #wandb.log({'examples': wandb_list}, commit=False)

                # generate masks
                # obs_map = mask_generator.generate(obs_img)
                # goal_map = mask_generator.generate(goal_img)

                #ax[2].imshow(obs_map['segmentation'])
                #ax[3].imshow(goal_map['segmentation'])

                #show_anns(obs_map, ax[0])
                #show_anns(goal_map, ax[1])
                
                show_anns(obs_map, ax[2])
                show_anns(goal_map, ax[3])

                ax[0].set_title('Observation')
                ax[1].set_title('Goal')
                ax[2].set_title('Obs Map')
                ax[3].set_title('Goal Map')

                for a in ax.flatten():
                    a.axis('off')

                map_save_path = os.path.join(visualize_path, f'maps_{i}.png')
                plt.savefig(map_save_path)

                wandb_list.append(wandb.Image(map_save_path))

                wandb.log({'examples': wandb_list}, commit=False)

                print(f"Finished generating masks for maps_{i}.png.")
"""
