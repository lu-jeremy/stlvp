""" PC Generation + Obj Detection"""

from ultralytics import YOLO
from ultralytics.data.dataset import YOLODataset

model = YOLO("../../drive/MyDrive/Purdue/CoRAL/Testing/models/yolov8n.pt")

dataset_dir = '../../dataset'

for root_, dir_, files_ in os.walk(dataset_dir):  # different dataset collection for each day
  date_bags = dir_
  break

# convert depth images to 3D point clouds

f = 1.3 # mm

IMG_SIZE = (128, 256)

# camera intrinsic matrix; requires camera specs
Q = np.array([
    [f, 0, IMG_SIZE[1] / 4],
    [0, f, IMG_SIZE[0] / 2],
    [0, 0, 1]
])

Q_inv = np.linalg.inv(Q)

"""### PC-Box Pairing"""

# point_cloud = []
# print(date_dirs)

def generate_point_cloud(img, ax, center_width=None, spherical=False):
  """
  Generates point cloud for fisheye and spherical camera lenses.

  Args:
    img: depth image
    ax: plot axis
    center_width: center of image for spherical images
    spherical: camera type

  Returns:
    NumPy array of 3D point clouds matching the number of pixels.
  """

  center_width = img.shape[1] // 2

  def extract_pc(img, spherical=spherical):
    curr_pc = np.zeros(shape=(img.shape[0] * img.shape[1], 3))
    print(img.shape)

    for v in range(img.shape[0]):
      for u in range(img.shape[1]):
        # print(v, u)
        depth = img[v, u]

        if spherical:
          normalized_x = (u - Q[0, 2]) / Q[0, 0]
          normalized_y = (v - Q[1, 2]) / Q[1, 1]
          theta = np.arctan2(normalized_x, 1)
          phi = np.arctan2(normalized_y, 1)
          # theta = (2 * np.pi * u) / img.shape[1]
          # phi = (np.pi * v) / img.shape[0]

          # convert spherical coordinates to 3D coordinates
          X = depth * np.sin(phi) * np.cos(theta)
          Y = depth * np.sin(phi) * np.sin(theta)
          Z = depth * np.cos(phi)

          point = np.array([X, Y, Z])
          # point = np.dot(Q_inv, [u, v, 1]) * depth
        else:
          point = np.dot(Q_inv, [u, v, 1]) * depth

        curr_pc[u * len(img) + v] = point

    return curr_pc

  if spherical:
    # cropped for both perspectives
    left = img[:, :center_width]
    right = img[:, center_width:]

    pc_left = extract_pc(left)
    # pc = pc_left  # TEMP
    pc_right = extract_pc(right)

    pc_right[:, 0] *= -1  # just flip the x-axis

    pc = np.vstack((pc_left, pc_right))
  else:
    # implement fisheye and other camera types later
    pc = extract_pc(img)

  return pc


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


