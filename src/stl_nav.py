import torch
import torch.nn.functional as F
import sys
import wandb

sys.path.insert(0, "stlcg/src")  # disambiguate path names

import stlcg
import stlviz as viz

from ultralytics import YOLO, SAM
from ultralytics.data.dataset import YOLODataset
from mobile_sam import sam_model_registry, SamAutomaticMaskGenerator


class STLViz:
    def __init__(
        self,
        unshuffled_dataset: torch.utils.data.DataLoader,
        weight_path: str = "weights/weight/sam_vit_h_4b8939.pth"  # TO-DO: put the correct path
    ):
        self.formula = self.generate_waypoints(unshuffled_dataset)
    
        # specify ViT mask generators 
        sam_chkpt = os.path.join(base_path, weight_path)

        model_type = 'vit_h'
        mobile_sam = sam_model_registry[model_type](checkpoint=sam_chkpt).to(device=device)
        mobile_sam.eval()

        self.mask_generator = SamAutomaticMaskGenerator(mobile_sam)

    def generate_waypoints(self, dataset) -> stlcg.STL_Formula:
        for i, data in enumerate(dataset):
            obs_img, goal_img, _, _, _, _, _ = data

            obs_img = TF.resize()

            map_ious = [obs["predicted_iou"] for obs in obs_map]
            avg_iou = sum(map_ious) / len(map_ious)

            #o = stlcg.Expression('obs', obs)
            #g = stlcg.Expression('g', goal)

            #intersection = (obs & goal).float().sum((1, 2)) + 1e-6
            #union = (obs | goal).float().sum(i(1, 2)) + 1e-6
            #iou = intersection / union
            iou = avg_iou * 2. - 1.  # without normalization set to >.5

            return stlcg.Always(stlcg.Expression('iou', iou))

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

