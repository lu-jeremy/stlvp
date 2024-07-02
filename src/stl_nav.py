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

from mobile_sam import sam_model_registry, SamAutomaticMaskGenerator
from vit_pytorch.mobile_vit import MobileViT
sys.path.insert(0, 'visualnav-transformer/train')
from vint_train.data.data_utils import VISUALIZATION_IMAGE_SIZE
from vint_train.visualizing.visualize_utils import to_numpy, from_numpy

sys.path.insert(0, "stlcg/src")  # disambiguate path names
import stlcg
import stlviz as viz


sam_dir = "pretrained_weights/weight/sam_vit_h_4b8939.pth"

mbvit_config = {
    'image_size': (256, 256),
    'dims': [96, 120, 144],
    'channels': [16, 32, 48, 48, 64, 64, 80, 80, 96, 96, 384],
    #'channels': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], 
    'num_classes': 1000,
}


def load_vit_models(device: torch.device) -> Tuple[SamAutomaticMaskGenerator, MobileViT]: 
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


def generate_masks(viz_obs: torch.Tensor, model) -> dict:
    viz_obs = np.moveaxis(to_numpy(viz_obs), 0, -1)
    viz_obs = (viz_obs * 255.).astype("uint8")
    return model.generate(viz_obs)


def mask_to_img(masks: dict) -> np.ndarray:
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


def generate_embeddings_from_obs(
    viz_obs: torch.Tensor,
    mb_sam: SamAutomaticMaskGenerator,
    mb_vit: MobileViT,
) -> torch.Tensor:
    # generate masks
    viz_obs = np.moveaxis(to_numpy(viz_obs), 0, -1)
    viz_obs = (viz_obs * 255.).astype("uint8")
    masks = mb_sam.generate(viz_obs)

    # preprocess mask images to be encoded
    mask_img = mask_to_img(masks)
   
    # encode mask image
    z_t = mb_vit(mask_img)

    return z_t


def generate_waypoints(
    dataset: DataLoader, 
    mb_sam: SamAutomaticMaskGenerator,
    mb_vit: MobileViT,
) -> Tuple[stlcg.STL_Formula, torch.Tensor]:
    z_stacked = None

    # load last latent file if not already saved
    latent_dir = "latents"
    latent_file = os.path.join(latent_dir, "last_saved_100x1000.pt")

    if not os.path.isdir(latent_dir):
        for i, data in enumerate(tqdm.tqdm(dataset, desc="Generating waypoints...")):
            obs_img, goal_img, _, _, _, _, _ = data

            obs_imgs = torch.split(obs_img, 3, dim=1)
            viz_obs_img = TF.resize(obs_imgs[-1], VISUALIZATION_IMAGE_SIZE[::-1])
            # TODO: implement goal images
            # viz_goal_img = TF.resize(goal_img, VISUALIZATION_IMAGE_SIZE[::-1])

            for obs in viz_obs_img:
                z_t = generate_embeddings_from_obs(
                        obs,
                        mb_sam,
                        mb_vit
                )
                
                if z_stacked is None:
                    z_stacked = z_t
                else:    
                    print("\nBEFORE")
                    print(z_stacked.shape)
                    print(z_t.shape)
                    z_stacked = torch.cat((z_stacked, z_t), dim=0)
                    print("AFTER\n", z_stacked.shape)

                # TODO: this is temporary
                if z_stacked.shape[0] == 100:
                    break
            break  # TODO: temp

        torch.save(z_stacked, latent_file)
    else:
        z_stacked = torch.load(latent_file)

    psi = None

    for z_t in z_stacked:
        cossim = 2 * torch.randn(1) - 1
        phi_t = stlcg.Expression('phi_t', cossim) > 0

        if psi is None:
            psi = phi_t
        else:
            psi = stlcg.Until(psi, phi_t)  # TODO: change to Until operator

    return psi, z_stacked


def compute_stl_loss(
    viz_obs: torch.Tensor, 
    z_embeddings: torch.Tensor, 
    formula: stlcg.STL_Formula,
    mb_sam: SamAutomaticMaskGenerator,
    mb_vit: MobileViT,
) -> torch.Tensor:
    stl_loss = 0
    margin = 0.05

    for i, obs in enumerate(viz_obs):
        z = generate_embeddings_from_obs(
                obs,
                mb_sam,
                mb_vit
        )
        
        inputs = []
        for z_t in z_embeddings:
            cossim = F.cosine_similarity(z_t, z, dim=1).float()
            inputs.append(cossim)

        inputs = torch.tensor(inputs)

        print("inputs==========", inputs)
        
        # STL loss
        if formula is None:
            raise ValueError(f"Formula is not properly defined: {self.formula}")

        robustness = (-formula.robustness(inputs)).squeeze() 
        stl_loss += F.leaky_relu(robustness - margin).mean()
    
    return stl_loss / viz_obs.shape[0]  # TODO: check if this is right


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
