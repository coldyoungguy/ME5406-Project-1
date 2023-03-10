"""
This contains the class for the environment for the robot to traverse in as well as a user interface to
interact and visualise each algorithm.
"""

import heapq
import time
import tkinter as tk
from tkinter import messagebox
from tkinter import ttk

from PIL import Image, ImageTk
from src.Algorithms.monte_carlo import MonteCarlo
from src.Algorithms.Q_learning import QLearning
from src.Algorithms.SARSA import SARSA
from src.Misc.schedulers import *

import numpy as np
from src.Misc.params import *

GREY = "#323232"        # Colour of the UI background
DARK_GREY = "#171717"   # Colour of the Grid background

# Colours used for the text in the console
class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

# Class for the environment
class Env(tk.Tk, object):
    def __init__(self):
        super(Env, self).__init__()

        # Creates the action space for the agent
        # in the order of right, left, down, up respectively
        self.action_space = [(0, 1), (0, -1), (1, 0), (-1, 0)]  # right, left, down, up

        # Initialises the starting state of the agent (0, 0)
        # Within params.py, can be changed to any other coordinates
        self.agent_state = START_COORD

        # Creates a blank list to store the path of the agent
        # should the path be the shortest seen, it would be stores in final_agent_path and
        # its length in shortest_agent_len.
        self.agent_path = []
        self.shortest_agent_len = float('inf')
        self.final_agent_path = []

        # Initialises learning rate, epsilon, gamma, number of episodes and obstacle weight
        # for use in start button resetting of parameters.
        self.learning_rate = LEARNING_RATE
        self.ep = EPSILON
        self.gamma = GAMMA
        self.episodes = NUM_EPISODES
        self.obstacle_weight = OBSTACLE_WEIGHT

        # For tuning purposes, if a fixed map is needed, USE_FIXED_MAP can be set to 4 or 7 or 10.
        # Otherwise, the map will be generated randomly and checked for solvability
        if USE_FIXED_MAP == 4:
            self.cellMap = np.array(FIXED_MAP_4)
            self.grid_size = 4
        elif USE_FIXED_MAP == 7:
            self.cellMap = np.array(FIXED_MAP_7)
            self.grid_size = 7
        elif USE_FIXED_MAP == 10:
            self.cellMap = np.array(FIXED_MAP_10)
            self.grid_size = 10
        else:
            self.grid_size = GRID_SIZE
            self.cellMap = self.generateMap()

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # The block of code below is for the user interface creation
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        # Setting the title and background colour of the UI
        self.title('Frozen Lake')
        self.configure(bg=GREY)

        # Used to generate the size of the grid map in terms of pixels
        self.map_size = 1000
        # The cell size in terms of pixels is then generated based off the map size and grid size
        self.cell_size = self.map_size // self.grid_size
        # Sets the FPS of the UI for easier visualisation if the updates becomes too fast
        self.fps = FPS
        # Stores the algorithm to be used, which cna be changed in the UI.
        self.method, self.lr_method, self.ep_method, self.gamma_method = '', '', '', ''
        # Sets up containers for the images to prevent them from being garbage collected
        # only properly initialised for efficiency.
        self.up_img, self.down_img, self.left_img, self.right_img = None, None, None, None
        # For faster runtimes, as during experimentation, the UI could slow down the algorithms
        self.runtime_updates = RUNTIME_UPDATES
        self.USE_GAMMA_SCHEDULE = tk.BooleanVar(USE_GAMMA_SCHEDULE)
        self.USE_EP_SCHEDULE = tk.BooleanVar(USE_EP_SCHEDULE)
        self.USE_LR_SCHEDULE = tk.BooleanVar(USE_LR_SCHEDULE)

        # Initialises dictionaries to store the images and numbers to prevent them from being garbage collected
        self.image_map = {}
        self.numbers_map = {}

        # Initialises the UI
        self.main_frame = tk.Frame(self, bg=GREY)
        self.main_frame.pack()

        # Creates the main separation between the top and bottom frames
        # Top frame storing settings and parameters
        # Bottom frame storing the map
        self.top_frame = tk.Frame(self.main_frame, bg=GREY)
        self.top_frame.grid(row=0, column=0, sticky="news")
        self.bottom_frame = tk.Frame(self.main_frame, bg=GREY)
        self.bottom_frame.grid(row=1, column=0, sticky="news")

        self.mapGen_frame = tk.LabelFrame(self.top_frame, text="Map Generation", bg=GREY)
        self.mapGen_frame.grid(row=1, column=0)

        self.mapGen_button = tk.Button(self.mapGen_frame, text="Regenerate", command=self.regenMap)
        self.mapGen_button.grid(row=0, column=0, columnspan=2)

        self.mapGen_mapSize_label = tk.Label(self.mapGen_frame, text="Map Size: ", bg=GREY, fg='white')
        self.mapGen_mapSize_label.grid(row=1, column=0)
        self.mapGen_mapSize = tk.Spinbox(self.mapGen_frame, from_=4, to=25)
        self.mapGen_mapSize.grid(row=1, column=1)

        self.mapGen_hole_label = tk.Label(self.mapGen_frame, text="Hole Chance", bg=GREY, fg='white')
        self.mapGen_hole_label.grid(row=2, column=0)
        self.mapGen_hole = tk.Spinbox(self.mapGen_frame, from_=0, to=1, format="%1.2f", increment=0.01,
                                      textvariable=tk.StringVar(self).set("0.30"))
        self.mapGen_hole.grid(row=2, column=1)

        self.settings_frame = tk.LabelFrame(self.top_frame, text="Settings", bg=GREY)
        self.settings_frame.grid(row=0, column=0)
        self.settings_label = tk.Label(self.settings_frame, text="FPS", bg=GREY, fg='white')
        self.settings_label.grid(row=0, column=0)
        self.settings_fps = tk.Spinbox(self.settings_frame, from_=4, to=10000, )
        self.settings_fps.grid(row=0, column=1)

        self.param_frame = tk.LabelFrame(self.top_frame, text="Parameters", bg=GREY)
        self.param_frame.grid(row=0, column=1, sticky="news", rowspan=2)

        self.param_method_label = tk.Label(self.param_frame, text="Method: ", bg=GREY, fg='white')
        self.param_method_label.grid(row=0, column=0)
        self.param_method = ttk.Combobox(self.param_frame, values=["Monte-Carlo", "Q-Learning", "SARSA"])
        self.param_method.grid(row=0, column=1)

        self.param_episodes_label = tk.Label(self.param_frame, text="Episodes: ", bg=GREY, fg='white')
        self.param_episodes_label.grid(row=1, column=0)
        self.param_episodes = tk.Spinbox(self.param_frame, from_=5, to=1000000)
        self.param_episodes.grid(row=1, column=1)

        self.param_epsilon_label = tk.Label(self.param_frame, text="Epsilon: ", bg=GREY, fg='white')
        self.param_epsilon_label.grid(row=2, column=0)
        self.param_epsilon = tk.Spinbox(self.param_frame, from_=0, to=100, format="%1.2f", increment=0.01)
        self.param_epsilon.grid(row=2, column=1)

        self.param_discount_label = tk.Label(self.param_frame, text="Discount: ", bg=GREY, fg='white')
        self.param_discount_label.grid(row=3, column=0)
        self.param_discount = tk.Spinbox(self.param_frame, from_=0, to=100, format="%1.2f", increment=0.01)
        self.param_discount.grid(row=3, column=1)

        self.param_lr_label = tk.Label(self.param_frame, text="Learning Rate: ", bg=GREY, fg='white')
        self.param_lr_label.grid(row=4, column=0)
        self.param_lr = tk.Spinbox(self.param_frame, from_=0, to=100, format="%1.2f", increment=0.01)
        self.param_lr.grid(row=4, column=1)

        self.paramButtons_frame = tk.Frame(self.param_frame, bg=GREY)
        self.paramButtons_frame.grid(row=5, column=0, columnspan=3)
        self.param_save = tk.Button(self.paramButtons_frame, text="Save", command=self.save)
        self.param_save.grid(row=0, column=0)
        self.param_start = tk.Button(self.paramButtons_frame, text="Start", command=self.start)
        self.param_start.grid(row=0, column=2)

        self.adv_settings_frame = tk.LabelFrame(self.top_frame, text="Advanced Settings", bg=GREY)
        self.adv_settings_frame.grid(row=0, column=2)

        self.lr_sch_check = tk.Checkbutton(self.adv_settings_frame, variable=self.USE_LR_SCHEDULE, bg=GREY)
        self.lr_sch_check.grid(row=0, column=0)
        self.lr_sch_label = tk.Label(self.adv_settings_frame, text="Learning Rate Scheduler", bg=GREY, fg='white')
        self.lr_sch_label.grid(row=0, column=1)
        self.lr_sch_combo = ttk.Combobox(self.adv_settings_frame,
                                         values=list(schedule_set.keys()))
        self.lr_sch_combo.grid(row=0, column=2, columnspan=2)

        self.gamma_sch_check = tk.Checkbutton(self.adv_settings_frame, variable=self.USE_GAMMA_SCHEDULE, bg=GREY)
        self.gamma_sch_check.grid(row=1, column=0)
        self.gamma_sch_label = tk.Label(self.adv_settings_frame, text="Discount Rate Scheduler", bg=GREY, fg='white')
        self.gamma_sch_label.grid(row=1, column=1)
        self.gamma_sch_combo = ttk.Combobox(self.adv_settings_frame,
                                         values=list(schedule_set.keys()))
        self.gamma_sch_combo.grid(row=1, column=2, columnspan=2)

        self.ep_sch_check = tk.Checkbutton(self.adv_settings_frame, variable=self.USE_EP_SCHEDULE, bg=GREY)
        self.ep_sch_check.grid(row=2, column=0)
        self.ep_sch_label = tk.Label(self.adv_settings_frame, text="Epsilon Scheduler", bg=GREY, fg='white')
        self.ep_sch_label.grid(row=2, column=1)
        self.ep_sch_combo = ttk.Combobox(self.adv_settings_frame,
                                         values=list(schedule_set.keys()))
        self.ep_sch_combo.grid(row=2, column=2, columnspan=2)

        self.map_frame = tk.LabelFrame(self.bottom_frame, text="Generated Map", bg=GREY, fg='white')
        self.map_frame.grid(row=1, column=0)
        self.map_widget = tk.Canvas(self.map_frame, bg=GREY,
                                    width=self.map_size,
                                    height=self.map_size)
        self.map_widget.grid(row=0, column=0)
        self.map_rects = {}

        self.createEnv()
        for widget in self.main_frame.winfo_children():
            for child in widget.winfo_children():
                child.grid_configure(padx=5, pady=5)
                child.configure(fg="white")

        print('Environment Initialised')

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Main Environment Creation with solvability checking
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    # If updates in the UI is disabled in params.py, the update function is overridden to do nothing
    # Else it updates the UI with the necessary UI elements
    def update(self):
        if not self.runtime_updates:
            return
        super(Env, self).update()

    # If a fixed map is not used, a random map is generated based on the chance of holes set by the user
    # The map is then checked for solvability using Dijkstra's algorithm
    # If the map is unsolvable, a new map is generated until a solvable map is found
    def generateMap(self):
        out_map = np.random.choice((1, 0),
                                   size=(self.grid_size, self.grid_size),
                                   p=(self.obstacle_weight, 1 - self.obstacle_weight))
        out_map[START_COORD[0], START_COORD[1]] = 0
        out_map[self.grid_size - 1, self.grid_size - 1] = 0
        while not self.dijkstra(out_map, START_COORD):
            print('Map is invalid: No possible path found')
            out_map = np.random.choice((1, 0),
                                       size=(self.grid_size, self.grid_size),
                                       p=(self.obstacle_weight, 1 - self.obstacle_weight))
            out_map[START_COORD[0], START_COORD[1]] = 0
            out_map[self.grid_size - 1, self.grid_size - 1] = 0

        print('Valid path found')
        print(out_map)
        return out_map

    # Dijkstra's algorithm is used to check if a path exists from the start to the goal
    # This simply returns True if a path exists, and False if no path exists
    def dijkstra(self, out_map, start):
        graph = {}
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                key = []
                if i != 0 and out_map[i - 1, j] != 1:
                    key.append((i - 1, j))
                if i != self.grid_size - 1 and out_map[i + 1, j] != 1:
                    key.append((i + 1, j))
                if j != self.grid_size - 1 and out_map[i, j + 1] != 1:
                    key.append((i, j + 1))
                if j != 0 and out_map[i, j - 1] != 1:
                    key.append((i, j - 1))
                graph[(i, j)] = key

        queue = [(0, start)]
        distances = {start: 0}
        visited = set()

        while queue:
            _, node = heapq.heappop(queue)
            if node in visited:
                continue
            visited.add(node)
            dist = distances[node]

            for neighbour in graph[node]:
                neighbour_dist = 1
                if neighbour in visited:
                    continue
                neighbour_dist += dist
                if neighbour_dist < distances.get(neighbour, float('inf')):
                    heapq.heappush(queue, (neighbour_dist, neighbour))
                    distances[neighbour] = neighbour_dist
        return True if (self.grid_size - 1, self.grid_size - 1) in distances.keys() else False

    # This function creates squares to represent cells in the grid based on the row and column
    def drawRec(self, r, c, colour):
        x1 = c * self.cell_size
        y1 = r * self.cell_size
        x2 = x1 + self.cell_size
        y2 = y1 + self.cell_size
        return self.map_widget.create_rectangle(x1, y1, x2, y2, fill=colour)

    # Once a valid map is generated, this function is called in the initialisation of the class
    # where it draws squares to represent the cells in the grid, white representing a hole,
    # while black represents a traversable cell.
    # A green outline is also added to the goal cell for easier visualisation
    def createEnv(self):
        print(f'Creating {self.grid_size}x{self.grid_size} grid')
        for r in range(self.grid_size):
            for c in range(self.grid_size):
                colour = DARK_GREY if self.cellMap[r][c] == 0 else 'white'
                self.map_rects[(r, c)] = self.drawRec(r, c, colour)
        self.map_widget.itemconfigure(self.map_rects[(self.grid_size - 1, self.grid_size - 1)], outline='green', width=3)

    # This function is called everytime the "Regenerate" button is clicked
    # This reinitialises all the variables needed to recreate a new map.
    def regenMap(self):
        self.grid_size = int(self.mapGen_mapSize.get())
        self.cell_size = self.map_size // self.grid_size
        self.obstacle_weight = float(self.mapGen_hole.get())
        self.cellMap = self.generateMap()
        self.map_widget.delete('all')
        self.createEnv()
        print('Map Regenerated')
        self.update()

    # Everytime an episode ends, this function is called to reset the environment
    # The agent is placed back at the starting position and the path is cleared
    def reset(self):
        self.agent_path = []
        self.agent_state = START_COORD
        self.map_widget.itemconfigure(self.map_rects[START_COORD], fill='red')
        self.update()
        return self.agent_state

    # draw_agent is called everytime the agent moves to a new cell
    # The previous cell is coloured back to its original colour using del_agent
    # while the new cell is coloured red representing the agent
    def del_agent(self, pos):
        colour = DARK_GREY if self.cellMap[pos[0]][pos[1]] == 0 else 'white'
        self.map_widget.itemconfigure(self.map_rects[pos], fill=colour)

    def draw_agent(self, old, new):
        self.del_agent(old)
        self.map_widget.itemconfigure(self.map_rects[new], fill='red')
        self.update()

    # At every new step, when the agent takes a new action, this function is called
    # The agent is moved to the new cell and the reward is calculated
    # If the agent reaches the goal, the episode ends and the reward is set to 1
    # If the agent falls into a hole, the episode ends and the reward is set to -1
    # If the agent moves to a traversable cell, the reward is set to 0
    # If the agent traverses out of the map, the episode ends and the reward is set to -inf
    def step(self, action):
        is_done = False
        current_state = (self.agent_state[0] + action[0], self.agent_state[1] + action[1])
        reward = 0

        if current_state[0] in range(self.grid_size) and current_state[1] in range(self.grid_size):
            self.draw_agent(self.agent_state, current_state)
            self.agent_state = current_state
            if self.cellMap[self.agent_state[0], self.agent_state[1]] == 0:
                self.agent_path.append(self.agent_state)
                if self.agent_state == (self.grid_size - 1, self.grid_size - 1):
                    # The agent is at the goal
                    reward = 1
                    is_done = True
                    print(f'{bcolors.OKGREEN}[INFO] Agent reached goal{bcolors.ENDC}')
                    self.del_agent(self.agent_state)
                    if len(self.agent_path) < self.shortest_agent_len:
                        self.shortest_agent_len = len(self.agent_path)
                        self.final_agent_path = self.agent_path

            elif self.cellMap[self.agent_state[0], self.agent_state[1]] == 1:
                # The agent has fell in hole
                print(f'{bcolors.WARNING}[INFO] Agent fell into hole{bcolors.ENDC}')
                self.del_agent(self.agent_state)
                reward = -1
                is_done = True
        else:
            # The agent fell off the map
            print('[INFO] Agent out of bounds')
            self.del_agent(self.agent_state)
            reward = float('-inf')
            is_done = True
        return self.agent_state, reward, is_done

    # def heatmap(self):

    # When the "Start" button is pressed, all the necessary variables is collected from the input boxes in the UI
    # The model is then initialised and the run function is called
    # Where the model used is also determined by the user.
    def start(self):
        self.fps = int(self.settings_fps.get())
        self.ep = float(self.param_epsilon.get())
        self.gamma = float(self.param_discount.get())
        self.learning_rate = float(self.param_lr.get())
        self.episodes = int(self.param_episodes.get())
        self.method = self.param_method.get()

        if self.method == '':
            error_box = messagebox.showerror(title='Model not set', message='Please select a method before continuing')
        elif self.lr_sch_combo.get() == '' and self.USE_LR_SCHEDULE:
            error_box = messagebox.showerror(title='Learning Rate schedule not set', message='Please select a method before continuing')
        elif self.gamma_sch_combo.get() == '' and self.USE_GAMMA_SCHEDULE:
            error_box = messagebox.showerror(title='Discount Rate schedule not set', message='Please select a method before continuing')
        elif self.ep_sch_combo.get() == '' and self.USE_EP_SCHEDULE:
            error_box = messagebox.showerror(title='Epsilon schedule not set', message='Please select a method before continuing')
        else:
            self.lr_method = schedule_set[self.lr_sch_combo.get()] if self.USE_LR_SCHEDULE else None
            self.gamma_method = schedule_set[self.gamma_sch_combo.get()] if self.USE_GAMMA_SCHEDULE else None
            self.ep_method = schedule_set[self.ep_sch_combo.get()] if self.USE_EP_SCHEDULE else None

            if self.method == 'Monte-Carlo':
                model = MonteCarlo(self, self.ep, self.gamma)
                model.run(self.episodes, gamma_schedule=self.gamma_method, ep_schedule=self.ep_method)
            elif self.method == 'Q-Learning':
                model = QLearning(self, self.ep, self.gamma, self.learning_rate)
                model.run(self.episodes, gamma_schedule=self.gamma_method, ep_schedule=self.ep_method, lr_schedule=self.lr_method)
            elif self.method == 'SARSA':
                model = SARSA(self, self.ep, self.gamma, self.learning_rate)
                model.run(self.episodes, gamma_schedule=self.gamma_method, ep_schedule=self.ep_method, lr_schedule=self.lr_method)

    # When training has completed, the final policy is drawn on the map
    # The policy is drawn by taking the action with the highest Q value for each state
    # and represented by arrows.
    # If a state has duplicate Q-values, it shows that there is no one optimal action
    # an arrow would not be drawn then and the cell would be left blank.
    def draw_final_policy(self, Q_Table):
        self.up_img = ImageTk.PhotoImage(Image.open('../Assets/up-arrow.png').resize((self.cell_size//2, self.cell_size//2), Image.ANTIALIAS))
        self.down_img = ImageTk.PhotoImage(Image.open('../Assets/down-arrow.png').resize((self.cell_size//2, self.cell_size//2), Image.ANTIALIAS))
        self.left_img = ImageTk.PhotoImage(Image.open('../Assets/left-arrow.png').resize((self.cell_size//2, self.cell_size//2), Image.ANTIALIAS))
        self.right_img = ImageTk.PhotoImage(Image.open('../Assets/right-arrow.png').resize((self.cell_size//2, self.cell_size//2), Image.ANTIALIAS))

        # Used to clean up the map before drawing the final policy
        # Glitches in the UI can happen when the FPS is too high and updates do not happen fast enough
        self.createEnv()

        for state in Q_Table.keys():

            # If the state is a hole or the goal, do not draw an arrow
            if self.cellMap[state[0], state[1]] == 1:
                continue
            if state[0] == self.grid_size -1 and state[1] == self.grid_size - 1:
                continue

            action_idx = np.argwhere(np.array(Q_Table[state]) == np.max(Q_Table[state]))
            if np.shape(action_idx)[0] > 1:
               continue
            else:
                action_idx = action_idx[0][0]

            if action_idx == 0: # Right
                self.image_map[state] = self.map_widget.create_image(
                    state[1] * self.cell_size + self.cell_size//2,
                    state[0] * self.cell_size + self.cell_size//2,
                    image=self.right_img)
            elif action_idx == 1: # Left
                self.image_map[state] = self.map_widget.create_image(
                    state[1] * self.cell_size + self.cell_size//2,
                    state[0] * self.cell_size + self.cell_size//2,
                    image=self.left_img)
            elif action_idx == 2: # Down
                self.image_map[state] = self.map_widget.create_image(
                    state[1] * self.cell_size + self.cell_size//2,
                    state[0] * self.cell_size + self.cell_size//2,
                    image=self.down_img)
            elif action_idx == 3: # Up
                self.image_map[state] = self.map_widget.create_image(
                    state[1] * self.cell_size + self.cell_size//2,
                    state[0] * self.cell_size + self.cell_size//2,
                    image=self.up_img)
        self.runtime_updates = True
        self.update()
        if not RUNTIME_UPDATES: self.runtime_updates = False

    # Draws the numbers on the map to check the number of times a cell is visited during training
    # Allows for analysis of if the final policy is based on luck or if it is optimal
    def draw_number(self, Numbers):
        for state in Numbers.keys():
            if self.cellMap[state[0], state[1]] == 1:
                continue
            if state[0] == self.grid_size - 1 and state[1] == self.grid_size - 1:
                continue

            self.numbers_map[state] = self.map_widget.create_text(
                state[1] * self.cell_size + self.cell_size//2,
                state[0] * self.cell_size + self.cell_size//2,
                text=str(Numbers[state]),
                fill='green',
                font='Helvetica 20 bold'
            )
        self.runtime_updates = True
        self.update()
        if not RUNTIME_UPDATES: self.runtime_updates = False


    # Saves the map as a png file
    def save(self, PATH=None):
        # Need to install Ghostscript for saving eps files
        if PATH is None:
            PATH = '../Results/'
        self.map_widget.postscript(file=f'{PATH}map.eps', colormode='color')
        self.update()
        print(f'Saving map to {PATH}')
        time.sleep(3)
        img = Image.open(f'{PATH}map.eps')
        img.save(f'{PATH}map.png', 'png', quality=100)


# For used when running this python file by itself.
if __name__ == "__main__":
    env = Env()
    env.mainloop()
