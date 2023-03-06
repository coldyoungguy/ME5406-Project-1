import heapq
import tkinter as tk
from tkinter import messagebox
from tkinter import ttk

from PIL import Image, ImageTk
from monte_carlo import MonteCarlo
from Q_learning import QLearning
from SARSA import SARSA

import numpy as np
from params import *

GREY = "#323232"
DARK_GREY = "#171717"

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


class Env(tk.Tk, object):
    def __init__(self):
        super(Env, self).__init__()
        self.title('Frozen Lake')
        self.configure(bg=GREY)

        self.map_size = 500
        self.grid_size = GRID_SIZE
        self.cell_size = self.map_size // GRID_SIZE
        self.action_space = [(0, 1), (0, -1), (1, 0), (-1, 0)]  # right, left, down, up
        self.agent_state = START_COORD
        self.agent_path = []
        self.shortest_agent_len = float('inf')
        self.final_agent_path = []
        self.fps = FPS
        self.learning_rate = LEARNING_RATE
        self.ep = EPSILON
        self.gamma = GAMMA
        self.episodes = NUM_EPISODES
        self.method = ''
        self.obstacle_weight = OBSTACLE_WEIGHT
        self.image_map = {}

        self.up_img, self.down_img, self.left_img, self.right_img = None, None, None, None

        if USE_FIXED_MAP == 4:
            self.cellMap = np.array(FIXED_MAP_4)
        elif USE_FIXED_MAP == 10:
            self.cellMap = np.array(FIXED_MAP_10)
        else: self.cellMap = self.generateMap()
        self.main_frame = tk.Frame(self, bg=GREY)
        self.main_frame.pack()

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
        self.param_test = tk.Button(self.paramButtons_frame, text="Test", command=self.test)
        self.param_test.grid(row=0, column=1)
        self.param_start = tk.Button(self.paramButtons_frame, text="Start", command=self.start)
        self.param_start.grid(row=0, column=2)

        self.map_frame = tk.LabelFrame(self.bottom_frame, text="Generated Map", bg=GREY, fg='white')
        self.map_frame.grid(row=1, column=0)
        self.map_widget = tk.Canvas(self.map_frame, bg=GREY,
                                    width=self.map_size,
                                    height=self.map_size)
        self.map_widget.grid(row=0, column=0)
        self.map_rects = {}

        self.log_label = tk.Label(self.bottom_frame, text="Log")
        # self.log = tk.T

        self.createEnv()
        for widget in self.main_frame.winfo_children():
            for child in widget.winfo_children():
                child.grid_configure(padx=5, pady=5)
                child.configure(fg="white")

        print('Environment Initialised')

    def createEnv(self):
        print(f'Creating {self.grid_size}x{self.grid_size} grid')
        for r in range(self.grid_size):
            for c in range(self.grid_size):
                colour = DARK_GREY if self.cellMap[r][c] == 0 else 'white'
                self.map_rects[(r, c)] = self.drawRec(r, c, colour)
        self.map_widget.itemconfigure(self.map_rects[(self.grid_size - 1, self.grid_size - 1)], outline='green', width=3)

    def drawRec(self, r, c, colour):
        x1 = c * self.cell_size
        y1 = r * self.cell_size
        x2 = x1 + self.cell_size
        y2 = y1 + self.cell_size
        return self.map_widget.create_rectangle(x1, y1, x2, y2, fill=colour)

    def regenMap(self):
        self.grid_size = int(self.mapGen_mapSize.get())
        self.cell_size = self.map_size // self.grid_size
        self.obstacle_weight = float(self.mapGen_hole.get())
        self.cellMap = self.generateMap()
        self.map_widget.delete('all')
        self.createEnv()
        print('Map Regenerated')
        self.update()

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

    def dijkstra(self, out_map, start):
        # Generate Neighbours
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

    def reset(self):
        self.agent_path = []
        self.agent_state = START_COORD
        self.map_widget.itemconfigure(self.map_rects[START_COORD], fill='red')
        self.update()
        return self.agent_state

    def del_agent(self, pos):
        colour = DARK_GREY if self.cellMap[pos[0]][pos[1]] == 0 else 'white'
        self.map_widget.itemconfigure(self.map_rects[pos], fill=colour)

    def draw_agent(self, old, new):
        self.del_agent(old)
        self.map_widget.itemconfigure(self.map_rects[new], fill='red')
        self.update()

    def step(self, action):
        is_done = False
        current_state = (self.agent_state[0] + action[0], self.agent_state[1] + action[1])
        reward = 0
        # print(self.agent_state, current_state)

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
                is_done = True
                reward = -1
        else:
            # The agent fell off the map
            print('[INFO] Agent out of bounds')
            self.del_agent(self.agent_state)
            is_done = True
            reward = float('-inf')
        return self.agent_state, reward, is_done

    # def heatmap(self):

    def start(self):
        self.fps = int(self.settings_fps.get())
        self.ep = float(self.param_epsilon.get())
        self.gamma = float(self.param_discount.get())
        self.learning_rate = float(self.param_lr.get())
        self.episodes = int(self.param_episodes.get())
        self.method = self.param_method.get()

        if self.method == '':
            error_box = messagebox.showerror(title='Model not set', message='Please select a method before continuing')
        else:
            if self.method == 'Monte-Carlo':
                model = MonteCarlo(self, self.ep, self.gamma)
                model.run(self.episodes)
            elif self.method == 'Q-Learning':
                model = QLearning(self, self.ep, self.gamma, self.learning_rate)
                model.run(self.episodes)
            elif self.method == 'SARSA':
                model = SARSA(self, self.ep, self.gamma, self.learning_rate)
                model.run(self.episodes)

    def draw_final_policy(self, Q_Table):
        self.up_img = ImageTk.PhotoImage(Image.open('../Assets/up-arrow.png').resize((self.cell_size//2, self.cell_size//2), Image.ANTIALIAS))
        self.down_img = ImageTk.PhotoImage(Image.open('../Assets/down-arrow.png').resize((self.cell_size//2, self.cell_size//2), Image.ANTIALIAS))
        self.left_img = ImageTk.PhotoImage(Image.open('../Assets/left-arrow.png').resize((self.cell_size//2, self.cell_size//2), Image.ANTIALIAS))
        self.right_img = ImageTk.PhotoImage(Image.open('../Assets/right-arrow.png').resize((self.cell_size//2, self.cell_size//2), Image.ANTIALIAS))
        for state in Q_Table.keys():
            if self.cellMap[state[0], state[1]] == 1:
                continue
            if state[0] == self.grid_size -1 and state[1] == self.grid_size - 1:
                continue

            max_ = max(Q_Table[state])
            if type(max_) == list:
               pass
            else:
                action_idx = Q_Table[state].index(max_)
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
        self.update()


    def save(self):
        print('Saving.....')
    def test(self):
        print('Testing Started')





if __name__ == "__main__":
    env = Env()
    env.mainloop()
