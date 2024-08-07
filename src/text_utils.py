import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from typing import Tuple, List, Optional
from collections import namedtuple
import gc

from constants import *

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


def valid_landmark(preds_unique: np.ndarray, num_thresh: int, object_landmarks: List) -> bool:
    """
    Images must have the correct object landmarks, have enough pixel
    predictions, and have enough landmarks.
    """
    # object_landmarks = [2, 3, 4, 5, 6, 7, 11, 12, 13, 14, 15, 16, 17, 18]
    # object_landmarks = np.array([2, 3, 4, 5, 6, 7])
    in_landmarks = np.intersect1d(preds_unique, object_landmarks, assume_unique=True)

    # skip if none of the conditions satisfy
    # return len(preds_unique) <= N and \
    # len(in_landmarks) != 0
    return len(in_landmarks) != 0


def filter_preds(
        outputs: np.ndarray,
        actions: Optional[torch.Tensor] = None,
        data_pos: int = 0,
) -> Tuple[List, List, np.ndarray]:
    """
    Filters predictions based on criteria, used for text-to-image subgoal generation.

    Args:
        outputs (`np.ndarray`):
            Raw outputs of semantic segmentation model.
        actions (`torch.Tensor`, *optional*):
            Ground-truth 2D action trajectories for the current run.
        data_pos (`int`):
            Dataset position in loading loop.

    Returns:
        `tuple`:
            Contains a list of prompts for the text-to-image model, corresponding intervals for each subgoal, and
            a list of trajectories for each waypoint.
    """
    # in the case that trajs are not used
    if actions is None:
        actions = []

    # return the max probability for each image
    preds = outputs.max(1)[1].detach().cpu().numpy()
    preds[preds == 255] = 19
    # format of image doesn't matter, flatten
    preds = preds.reshape(preds.shape[0], -1)

    preds_comb = [np.unique(p, return_counts=True)
                  for p in preds]

    # filter out predictions
    trajs = []
    preds = []
    intervals = []

    # object persistence variables
    curr_pred = None
    curr_lm = None
    start = None
    end = None
    # time elapsed between initial tracking of landmark
    dt = 0
    # valid landmarks
    object_landmarks = np.array([2, 3, 4, 5, 6, 7])
    # object_landmarks = np.array([i for i in range(19)])

    # retrieve n-top unique classes
    for t, (preds_unique, counts) in enumerate(preds_comb):
        # shift based on dataset loop idx
        t_start = data_pos + t

        lm_id = preds_unique[np.argmax(counts)]

        # skip preds based on criteria
        # if not valid_landmark(preds_unique, NUM_THRESH, object_landmarks):

        # we only care about the main landmark
        if lm_id not in object_landmarks:
            # the only case where we wouldn't increase the interval with a wrong landmark is before starting
            if start is not None:
                end = start + dt

                # only when a valid landmark is seen and robot sees objects for more than dt frames
                if not dt < PERSISTENCE_THRESH:
                    print(f"Not good landmark -> final interval: [{start}, {end}]")
                    # if robot sees object for only a couple frames, it's not good grounds to include it as a landmark
                    intervals.append([start, end])
                    preds.append(curr_pred)
                    trajs.append(actions[start: end + 1])

                    assert intervals[0][1] - intervals[0][0] == len(trajs[0]) - 1
                    # preds.append(curr_lm)  # for now, add full prediction after filtering

                # next iteration, update with an valid landmark or skip
                curr_lm = None
                start = None
                dt = 0
            # print("SKIP")
            continue

        # filter out unique predictions
        # don't do unnecessary computation if not necessary
        # if preds_unique.shape[0] < NUM_THRESH:
        # top_idx = np.arange(len(preds_unique))
        # else:
        # top_idx = heapq.nlargest(NUM_THRESH, preds_unique)  # for sets
        # top_idx = np.argpartition(preds_unique, -NUM_THRESH)[-NUM_THRESH:]
        # n_top = preds_unique[top_idx]

        # handle object persistence: top-1 landmark
        # lm_id = preds_unique[np.argmax(counts)]

        # sometimes unique predictions per img are limited
        # if preds_unique.shape[0] < NUM_THRESH:
        #     top_idx = np.arange(len(preds_unique))
        # else:
        #     top_idx = np.argpartition(counts, -NUM_THRESH)[-NUM_THRESH:]
            # print("Top idx based on counts", top_idx)
            # print("Counts", counts)
        # lm_id = preds_unique[top_idx]  # subset of original predictions

        # lm_id = preds_unique
        # # print("LM ID", lm_id)
        # mask = np.zeros_like(lm_id, dtype=bool)
        # for lm in object_landmarks:
        #     mask |= (lm_id == lm)
        # # print("MASK", mask)
        # lm_id = lm_id[mask]  # only take a subset of objects
        #
        # # if there aren't any objects in the subset, continue
        # if not np.any(lm_id):
        #     dt += 1
        #     continue

        # print(f"Current landmark: {curr_lm}, Updated lm: {lm_id}")

        if curr_lm is None:
            curr_lm = lm_id
            curr_pred = preds_unique
            start = t_start
            continue

        # if len(np.intersect1d(lm_id, curr_lm, assume_unique=True)) == 0:
        if lm_id != curr_lm:
            # print("New landmark found \n")
            end = start + dt

            # if object persistence < dt, restart the counter, but the current landmark doesn't qualify
            if not dt < PERSISTENCE_THRESH:
                print(f"Good landmark -> final interval: [{start}, {end}]")
                intervals.append([start, end])
                preds.append(curr_pred)
                trajs.append(actions[start: end + 1])

                assert intervals[0][1] - intervals[0][0] == len(trajs[0]) - 1

            # update current landmark if they're different
            curr_lm = lm_id
            curr_pred = preds_unique
            # only update start at the beginning of each new landmark
            start = t_start
            dt = 0
            # print(f"Start: {start}\n")
        else:
            dt += 1
            # print(f"Time elapsed: {dt} \n")

    intervals = np.array(intervals)
    # print("INTERVAL SHAPE", intervals.shape)

    # convert to class labels based on indices
    preds = [train_id_to_name[p] for p in preds]

    preds = [" and ".join(preds[i].reshape(-1)) \
                for i in range(len(preds))
    ]

    print("PROMPTS:", preds)
    print(f"NUM IMAGES/PROMPTS: {len(preds)}")

    return preds, trajs, intervals


def visualize_segmentation(obs_imgs: torch.Tensor, unnormalized_preds: np.ndarray):
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


def flush():
    """
    Frees up heap + GPU cached memory
    """
    gc.collect()
    torch.cuda.empty_cache()