import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torch.utils.data import DataLoader
import sys
import wandb
from typing import Tuple
import numpy as np
import matplotlib.pyplot as plt
import os
import tqdm
import cv2
import warnings

# be cautious of when real warnings appear
warnings.filterwarnings("ignore")

from mobile_sam import sam_model_registry, SamAutomaticMaskGenerator
from vit_pytorch.mobile_vit import MobileViT

sys.path.insert(0, 'visualnav-transformer/train')
from vint_train.data.data_utils import VISUALIZATION_IMAGE_SIZE
from vint_train.visualizing.visualize_utils import to_numpy, from_numpy

sys.path.insert(0, "stlcg/src")  # disambiguate path names
import stlcg
import stlviz as viz

"""
TODO
- Ensure that we don't need SAM
- 

"""

sam_dir = "pretrained_weights/weight/sam_vit_h_4b8939.pth"

mbvit_config = {
    'image_size': (256, 256),
    'dims': [96, 120, 144],
    'channels': [16, 32, 48, 48, 64, 64, 80, 80, 96, 96, 384],
    #'channels': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], 
    'num_classes': 1000,
}


def load_vit_models(device: torch.device) -> Tuple[SamAutomaticMaskGenerator, MobileViT]: 
    """
    Loads the pre-trained ViT models onto the specified device.
    """

    # specify ViT mask generators 
    weight_path = os.path.join(os.getcwd(), sam_dir)
    model_type = 'vit_h'
    model_class = sam_model_registry[model_type](checkpoint=weight_path).to(device=device).eval()
    mb_sam = SamAutomaticMaskGenerator(model_class)

    # specify ViT image encoder
    mb_vit = MobileViT(
            image_size=mbvit_config['image_size'],
            dims=mbvit_config['dims'],
            channels=mbvit_config['channels'],
            num_classes=mbvit_config['num_classes'],
    )

    return mb_sam, mb_vit
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
    viz_obs_img: torch.Tensor,
    mb_sam: SamAutomaticMaskGenerator,
    mb_vit: MobileViT,
) -> torch.Tensor:
    """
    Generates latents from the observation needed for STL.
    """

    # generate masks
    # viz_obs = np.moveaxis(to_numpy(viz_obs), 0, -1)
    # viz_obs = (viz_obs * 255.).astype("uint8")
    # masks = mb_sam.generate(viz_obs)

    # preprocess mask images to be encoded
    # mask_img = mask_to_img(masks)
   
    # encode mask image
    # z_t = mb_vit(mask_img)

    # print("shape before passing to vit", viz_obs_img.shape)

    # the whole point is to process the images in parallel
    z_t = mb_vit(viz_obs_img)

    return z_t


def generate_waypoints(
    dataset: DataLoader, 
    mb_sam: SamAutomaticMaskGenerator,
    mb_vit: MobileViT,
    visualize_waypoint: bool=False,
) -> torch.Tensor:
    """
    Generates latent waypoints from the full, unshuffled dataset.
    """

    z_stacked = None

    # load last latent file if not already saved
    latent_dir = "latents"
    latent_file = os.path.join(latent_dir, "last_saved.pt")

    # ensure directory is created
    if not os.path.isdir(latent_dir):
        os.makedirs(latent_dir)

    if not os.path.isfile(latent_file):
        for i, data in enumerate(tqdm.tqdm(dataset, desc="Generating waypoints...")):
            obs_img, goal_img, _, _, _, _, _ = data

            #print("0 shape", obs_img.shape)
            obs_imgs = torch.split(obs_img, 3, dim=1)

            # concatenate everything together (not needed)
            # obs_imgs = torch.cat(obs_imgs, dim=0)

            # reshape to MobileViT-compatible shape
            viz_obs_imgs = TF.resize(obs_imgs[-1], [256, 256])

            # save one of the visualization images (optional)
            if visualize_waypoint:
                save_path = "observation.png"

                viz_obs_img = np.moveaxis(to_numpy(viz_obs_imgs[0]), 0, -1)
                plt.imshow(viz_obs_img)
                plt.savefig(save_path)

                wandb.log({"ex": [wandb.Image(save_path)]}, commit=False)

            # TODO: implement goal images
            # viz_goal_img = TF.resize(goal_img, VISUALIZATION_IMAGE_SIZE[::-1])

            # convert the resized batch into latent vectors
            z_w = generate_latents_from_obs(
                    viz_obs_imgs,
                    mb_sam,
                    mb_vit
            )

            print(f"shape of the stack: {z_t.shape}")

            # recursively concatenate latent vectors
            if z_stacked is None:
                z_stacked = z_w
            else:
                z_stacked = torch.cat((z_stacked, z_w), dim=0)

            # TODO: temporary stop point
            if z_stacked.shape[0] == 256:
                break

            # for obs in viz_obs_img:
            #     print("Generating embeddings from observation...")
            #     z_t = generate_embeddings_from_obs(
            #             obs,
            #             mb_sam,
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

        torch.save(z_stacked, latent_file)
        print("Successfully saved waypoint latents!")
    else:
        z_stacked = torch.load(latent_file)
        print("Successfully loaded waypoint latents!")

    return z_stacked


def compute_stl_loss(
    viz_obs: torch.Tensor, 
    waypoint_latents: torch.Tensor, 
    # formula: stlcg.STL_Formula,
    mb_sam: SamAutomaticMaskGenerator,
    mb_vit: MobileViT,
    visualize_formula: bool=False,
) -> torch.Tensor: 
    """
    Generates STL formula and inputs for robustness, while computing the STL
    loss based on robustness.
    """
    inputs = []
    obs_latents = generate_latents_from_obs(viz_obs, mb_sam, mb_vit)
    
    # for each waypoint, compute cos_sim for the observation batch
    outer_form = None
    # indicates the sensitivity of satisfaction for cosine distance
    THRESHOLD = 0.2

    """ Method 1:
    O(nm) time
    """

    # print(waypoint_latents.shape, obs_latents.shape)

    # ensure that all latent observations are close to at least one waypoint latent
    # for i in range(waypoint_latents.shape[0]):
    #     inner_form = None
    #     for j in range(obs_latents.shape[0]):
    #         # print(i, j)
    #         # print(waypoint_latents[i].shape)
    #         # print(obs_latents[j].shape)
    #         cossim = F.cosine_similarity(waypoint_latents[i], obs_latents[j], dim=0).float()
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

    """ Method 2:
    O(n) with Until temporal operator
    """

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

    """ Method 3:
    O(m), process observation costs in parallel
    """

    outer_form = None
    THRESHOLD = 0.6  # TODO: verify that this works 

    for i in range(waypoint_latents.shape[0]):
        #TODO: temp
        if i == 3:
            break

        # process observation latents in parallel
        cossim = F.cosine_similarity(
                waypoint_latents[i].unsqueeze(0),
                obs_latents,
                dim=1
        )
        inputs.append(cossim.unsqueeze(-1).unsqueeze(-1))
        cossim = stlcg.Expression("phi_j", cossim) > THRESHOLD

        if outer_form is None:
            outer_form = cossim
        else:
            outer_form = outer_form | cossim

    #TODO: temp
    print(inputs[0].shape, inputs[1].shape, inputs[2].shape)
    inputs = ((inputs[0], inputs[1]), inputs[2])

    print("cossim", F.cosine_similarity(waypoint_latents[0].unsqueeze(0),
        torch.rand(waypoint_latents[0].shape)))

    # inputs = torch.tensor(inputs)
    # print("inputs==========", min(inputs[0]), max(inputs[0]))

    """ Method 4:
    O(m)
    """
     
    if outer_form is None:
        raise ValueError(f"Formula is not properly defined.")

    print(outer_form)

    # saves a digraph of the STL formula
    if visualize_formula:
        digraph = viz.make_stl_graph(outer_form)
        viz.save_graph(digraph, "utils/formula")
        print("Saved formula CG successfully.")

    # compute the robustness
    margin = 0.0
    robustness = (-outer_form.robustness(inputs)).squeeze() 
    stl_loss = F.leaky_relu(robustness - margin).mean()

    print("STL loss:", stl_loss)
    
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
