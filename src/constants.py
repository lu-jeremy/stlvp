IMG_DIR = './images'
WP_DIR = './waypoints'
LATENT_DIR = './latents'
TRAJ_DIR = './traj'

NUM_THRESH = 1
PERSISTENCE_THRESH = 10  # Δt for object persistence
SIM_THRESH = [5, 0.15]  # indicates the satisfaction threshold for similarity distance
WEIGHT_STL = 5e-3  # stl loss weight in full diffusion objective

LOAD_WAYPOINTS = True  # `True` if training, `False` if pre-processing
ENABLE_INTERVALS = True  # enable the use of intervals

RESET_IMG_DIR = False  # erase and re-create the image directory
RESET_LATENT_DIR = False  # erase and re-create the latents directory

PROCESS_SUBGOALS = True  # process dataset subgoals
PROCESS_GOALS = True  # process dataset goals

VISUALIZE_DATASET = False  # visualize example batch dataset images
VISUALIZE_SUBGOALS = False  # visualize the generated subgoal images
VISUALIZE_STL = False  # visualize the STL formula
VISUALIZE_SIM = False  # visualize the similarity metrics
VISUALIZE_TRAJ = True  # visualize waypoint and predicted trajectories in STL computation
VIS_FREQ = 50  # for similarity metrics
ANIM_FREQ = 1000  # for visualizing the trajectory animations
VISUALIZATION_IMAGE_SIZE = (640, 680)
