# Environment Parameters
OBSTACLE_WEIGHT = 0.1
FPS = 0
START_COORD = (0, 0)
USE_FIXED_MAP = 4
GRID_SIZE = 4
if USE_FIXED_MAP: GRID_SIZE = USE_FIXED_MAP
FIXED_MAP_4 = [[0, 0, 0, 0],
               [0, 1, 0, 1],
               [0, 0, 0, 0],
               [1, 0, 0, 0]]

FIXED_MAP_10 = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 0, 1, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                [1, 0, 1, 1, 0, 0, 0, 0, 0, 1],
                [0, 0, 0, 0, 0, 0, 1, 0, 1, 1],
                [0, 0, 0, 1, 0, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 1, 0, 0, 0, 0]]

# Algo Parameters
EPSILON = 0.5
GAMMA = 0.9
LEARNING_RATE = 0.1
NUM_STEPS = 100
NUM_EPISODES = 500
ACCURACY_RANGE = 20
ROLLING_AVG = 1000
USE_LR_SCHEDULE = False
USE_EP_SCHEDULE = False


# Plotting Parameters
PLOT_INDIVIDUAL = False
PLOT_LEARNING_RATE = True
PLOT_GAMMA = True

