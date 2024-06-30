import sys
sys.path.insert(0, 'stlcg/src')  # disambiguate path names
sys.path.insert(0, 'visualnav-transformer/train')
print(sys.path)

import stlcg
import stlviz as viz
from stlcg import Expression
from utils import print_learning_progress

from vint_train.process_data.process_data_utils import *

import torch
import numpy as np
import matplotlib.pyplot as plt
#import requests
import cv2
#import pickle
#import io
#import yaml
# import rosbag
# from bs4 import BeautifulSoup
# from PIL import Image

import os
import tqdm
import argparse

from ultralytics import YOLO
from ultralytics.data.dataset import YOLODataset

def generate_bb(img, spherical=False):
  if spherical:
    # account for left and right images

    central_width = img.shape[1] // 2
    left = img[:, :central_width]
    right = img[:, central_width:]
    results = model(left)
    results_ = model(right)

    results.extend(results_)
    del results_
  else:
    results = model(img)

  all_boxes = []  # records boxes for each obj in frame

  for i, result in enumerate(results):
    boxes = result.boxes.xyxy.cpu().numpy()
    if boxes.shape[0] == 0:
      continue
    for box in boxes:
      # found no boxes
      all_boxes.append(box)

  print('All boxes:', all_boxes)
  return results, np.array(all_boxes)  # must return both

def plot_bb(results, img):
    for result in results:
      boxes = result.boxes.xyxy.cpu().numpy()  # Bounding box coordinates
      confidences = result.boxes.conf.cpu().numpy()  # Confidence scores
      class_ids = result.boxes.cls.cpu().numpy()  # Class IDs

      for box, confidence, class_id in zip(boxes, confidences, class_ids):
          x1, y1, x2, y2 = map(int, box)
          label = f'{class_id}: {confidence:.2f}'
          color = (0, 255, 0)  # Green color for bounding box

          # Draw the bounding box
          cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
          cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.1, color, 1)

    # Display the image with bounding boxes
    cv2_imshow(img)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--yolo-path",
        "-y",
        default="./yolov8n.pt",
        type=str,
        help='YOLO pretrained chkpt'
    )

    parser.add_argument(
        "--input-dir",
        "-i",
        default="./data",
        type=str,
        help="path for input dataset"
    )

    args = parser.parse_args()

    model = YOLO(args.yolo_path)

    dataset_dir = args.input_dir

    for root_, dir_, files_ in os.walk(dataset_dir):  # different dataset collection for each day
      date_bags = dir_
      break

    IMG_SIZE = (128, 256)

    for date in date_bags:
      print(date)
      depth_saved_dir = os.path.join(dataset_dir, date, depth_subdir)
      img_saved_dir = os.path.join(dataset_dir, date, img_subdir)
      pc_dir = os.path.join(dataset_dir, date, 'point_clouds')
      bb_dir = os.path.join(dataset_dir, date, 'boxes')
      os.makedirs(pc_dir, exist_ok=True)
      os.makedirs(bb_dir, exist_ok=True)

      for root, dir, files in os.walk(depth_saved_dir):
        for i, file in enumerate(files):
          fig = plt.figure(figsize=(8, 8))
          ax = fig.add_subplot(111, projection='3d')

          # extract bounding boxes -> crop depth image -> point cloud
          img = cv2.imread(os.path.join(img_saved_dir, file))
          results, bounding_boxes = generate_bb(img, spherical=True)

          # np.save(os.path.join(bb_dir, f'{file[:file.index(".jpg")]}'), pc)

          plot_bb(results, img)

          depth_img = cv2.imread(os.path.join(root, file), cv2.IMREAD_ANYDEPTH)

          for box in bounding_boxes:
            x1, y1, x2, y2 = map(int, box)

            object_img = depth_img[y1: y2, x1: x2]

            cv2_imshow(object_img)

            pc = generate_point_cloud(object_img, ax, spherical=True)
            dist = np.sqrt(np.square(pc[:, 0]) + np.square(pc[:, 1]) + np.square(pc[:, 2]))
            ax.scatter(pc[:, 0], pc[:, 1], pc[:, 2], c=dist, cmap=plt.cm.plasma)

          # np.save(os.path.join(pc_dir, f'{file[:file.index(".jpg")]}'), pc)

          # ax.view_init(elev=0, azim=150)
          # ax.view_init(20, 120)
          # ax.view_init(elev=90, azim=0)
          plt.show()

          # sys.exit()

          cv2_imshow(depth_img)
          cv2_imshow(cv2.imread(os.path.join(root, '..', img_subdir, file)))
          # break

          if i == 50:
            import sys
            sys.exit()



"""

t = np.arange(-3, 3 , 0.2, dtype=np.float32)
w = torch.tensor(0.5 * np.exp(-t ** 2).reshape([1, t.shape[0], 1]), requires_grad=False)
x = torch.tensor(0.4 * np.exp(-(t + 0.5) ** 2).reshape([1, t.shape[0], 1]), requires_grad=False)

c = torch.tensor(1.0, dtype=torch.float, requires_grad=True)
d = torch.tensor(0.9, dtype=torch.float, requires_grad=True)

plt.figure(figsize=(10, 5))
plt.plot(t, w[0, :, 0], label='w')
plt.plot(t, x[0, :, 0], label='x')
plt.title('Robustness Traces')
plt.legend()

# allows us to overload PyTorch comparison operators
w_exp = Expression('x', w)
x_exp = Expression('y', x)
c_exp = Expression('c', c)
d_exp = Expression('d', d)

formula_one = w_exp <= c_exp
formula_two = x_exp >= d_exp
formula_three = stlcg.Always(w_exp >= (c_exp + d_exp))

formula = formula_three
# formula = stlcg.Until(formula_three, formula_one & formula_two)

digraph = viz.make_stl_graph(formula)
viz.save_graph(digraph, 'ex_cg')

inputs = w

formula(inputs, p_scale=1, scale=-1)

var_dict = {'c': c, 'd': d}
learning_rate = 0.05
device = torch.device('cuda')
optimizer = torch.optim.Adam(var_dict.values(), lr=learning_rate)
scale = 0.5

y = torch.tensor(-2 * np.arange(-3, 3, 0.2).reshape([1, 30, 1]), requires_grad=False)
z = torch.tensor(2 * np.arange(-3, 3, 0.2).reshape([1, 30, 1]), requires_grad=False)
y_exp = Expression('y', y)
z_exp = Expression('z', z)
v = torch.tensor(np.full(30, 1).reshape([1, 30, 1]), dtype=torch.float, requires_grad=True)

# inputs = (v, (y_exp, z_exp))
inputs = (w, (w_exp, x_exp))

l2_norm = torch.nn.MSELoss()
leaky_relu = torch.nn.LeakyReLU()

for i in range(2000):
  optimizer.zero_grad()  # reset gradient to prevent accumulation over multiple passes

  robustness = formula.robustness(inputs, scale=1)
  loss = leaky_relu(-robustness).mean()  # condense dim

  if i % 500:
    print_learning_progress(formula, inputs, var_dict, i, loss, 1)

  loss.backward()  # calculate and accumulate gradient
  optimizer.step()  # update model parameters with Adam

"""


