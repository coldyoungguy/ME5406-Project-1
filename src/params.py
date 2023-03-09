# Environment Parameters
OBSTACLE_WEIGHT = 0.1
FPS = 0
START_COORD = (0, 0)
USE_FIXED_MAP = 10
GRID_SIZE = 4
FIXED_MAP_4 = [[0, 0, 0, 0],
               [0, 1, 0, 1],
               [0, 0, 0, 0],
               [1, 0, 0, 0]]

FIXED_MAP_10_1 = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 0, 1, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                [1, 0, 1, 1, 0, 0, 0, 0, 0, 1],
                [0, 0, 0, 0, 0, 0, 1, 0, 1, 1],
                [0, 0, 0, 1, 0, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 1, 0, 0, 0, 0]]

FIXED_MAP_10 = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 0, 1, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 1, 0, 1, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 0, 1, 1, 0],
                [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 1, 1, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]

# Algo Parameters
EPSILON = 0.5
GAMMA = 0.9
LEARNING_RATE = 0.001
NUM_STEPS = 100
NUM_EPISODES = 10000
ACCURACY_RANGE = 20
ROLLING_AVG = 1000
USE_LR_SCHEDULE = False
USE_EP_SCHEDULE = False


# Plotting Parameters
PLOT_INDIVIDUAL = False
PLOT_LEARNING_RATE = True
PLOT_GAMMA = True

