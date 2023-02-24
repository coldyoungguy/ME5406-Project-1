import heapq
import tkinter as tk
from tkinter import ttk
import numpy as np
from params import *

GREY = "#323232"
DARK_GREY = "#171717"

class Env(tk.Tk, object):
    def __init__(self):
        super(Env, self).__init__()
        self.title('Frozen Lake')
        self.configure(bg=GREY)

        self.grid_size = GRID_SIZE  # 10
        self.cell_size = CELL_SIZE
        self.action_space = ['up', 'down', 'left', 'right']
        self.num_states = len(self.action_space)
        self.num_states = self.grid_size ** 2

        self.cellMap = self.generateMap()
        self.main_frame = tk.Frame(self, bg=GREY)
        self.main_frame.pack()

        self.top_frame = tk.Frame(self.main_frame, bg=GREY)
        self.top_frame.grid(row=0, column=0, sticky="news")
        self.bottom_frame = tk.Frame(self.main_frame, bg=GREY)
        self.bottom_frame.grid(row=1, column=0, sticky="news")

        self.mapGen_frame = tk.LabelFrame(self.top_frame, text="Map Generation", bg=GREY)
        self.mapGen_frame.grid(row=1, column=0)

        self.mapGen_button = tk.Button(self.mapGen_frame, text="Regenerate")
        self.mapGen_button.grid(row=0, column=0, columnspan=2)

        self.mapGen_mapSize_label = tk.Label(self.mapGen_frame, text="Map Size: ")
        self.mapGen_mapSize_label.grid(row=1, column=0)
        self.mapGen_mapSize = tk.Spinbox(self.mapGen_frame, from_=4, to=25)
        self.mapGen_mapSize.grid(row=1, column=1)

        self.mapGen_hole_label = tk.Label(self.mapGen_frame, text="Hole Possibility")
        self.mapGen_hole_label.grid(row=2, column=0)
        self.mapGen_hole = tk.Spinbox(self.mapGen_frame, from_=0, to=1, format="%1.2f", increment=0.01, textvariable=tk.StringVar(self).set("0.30"))
        self.mapGen_hole.grid(row=2, column=1)

        self.settings_frame = tk.LabelFrame(self.top_frame, text="Settings", bg=GREY)
        self.settings_frame.grid(row=0, column=0)
        self.settings_label = tk.Label(self.settings_frame, text="FPS")
        self.settings_label.grid(row=0, column=0)
        self.settings_fps = tk.Spinbox(self.settings_frame, from_=4, to=25)
        self.settings_fps.grid(row=0, column=1)

        self.param_frame = tk.LabelFrame(self.top_frame, text="Parameters", bg=GREY)
        self.param_frame.grid(row=0, column=1, sticky="news", rowspan=2)

        self.param_method_label = tk.Label(self.param_frame, text="Method: ")
        self.param_method_label.grid(row=0, column=0)
        self.param_method = ttk.Combobox(self.param_frame, values=["Mote-Carlo", "SARSA", "Q-Learning"])
        self.param_method.grid(row=0, column=1)

        self.param_episodes_label = tk.Label(self.param_frame, text="Episodes: ")
        self.param_episodes_label.grid(row=1, column=0)
        self.param_episodes = tk.Spinbox(self.param_frame, from_=5, to=1000000)
        self.param_episodes.grid(row=1, column=1)

        self.param_epsilon_label = tk.Label(self.param_frame, text="Epsilon: ")
        self.param_epsilon_label.grid(row=2, column=0)
        self.param_epsilon = tk.Spinbox(self.param_frame, from_=0, to=100, format="%1.2f", increment=0.01)
        self.param_epsilon.grid(row=2, column=1)

        self.param_discount_label = tk.Label(self.param_frame, text="Discount: ")
        self.param_discount_label.grid(row=3, column=0)
        self.param_discount = tk.Spinbox(self.param_frame, from_=0, to=100, format="%1.2f", increment=0.01)
        self.param_discount.grid(row=3, column=1)

        self.param_lr_label = tk.Label(self.param_frame, text="Learning Rate: ")
        self.param_lr_label.grid(row=4, column=0)
        self.param_lr = tk.Spinbox(self.param_frame, from_=0, to=100, format="%1.2f", increment=0.01)
        self.param_lr.grid(row=4, column=1)

        self.paramButtons_frame = tk.Frame(self.param_frame, bg=GREY)
        self.paramButtons_frame.grid(row=5, column=0, columnspan=3)
        self.param_save = tk.Button(self.paramButtons_frame, text="Save")
        self.param_save.grid(row=0, column=0)
        self.param_test = tk.Button(self.paramButtons_frame, text="Test")
        self.param_test.grid(row=0, column=1)
        self.param_start = tk.Button(self.paramButtons_frame, text="Start")
        self.param_start.grid(row=0, column=2)

        self.map_frame = tk.LabelFrame(self.bottom_frame, text="Generated Map", bg=GREY)
        self.map_frame.grid(row=1, column=0)
        self.map_widget = tk.Canvas(self.map_frame, bg=GREY,
                                    width=self.grid_size * self.cell_size,
                                    height=self.grid_size * self.cell_size)
        self.map_widget.grid(row=0, column=0)

        self.log_label = tk.Label(self.bottom_frame, text="Log")
        self.log = tk.T

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
                self.drawRec(r, c, colour)

    def drawRec(self, r, c, colour):
        x1 = c * self.cell_size
        y1 = r * self.cell_size
        x2 = x1 + self.cell_size
        y2 = y1 + self.cell_size
        self.map_widget.create_rectangle(x1, y1, x2, y2, fill=colour)

    def generateMap(self):
        out_map = np.random.choice((1, 0),
                                   size=(self.grid_size, self.grid_size),
                                   p=(OBSTACLE_WEIGHT, 1-OBSTACLE_WEIGHT))
        out_map[START_COORD[0], START_COORD[1]] = 0
        out_map[END_COORD[0], END_COORD[1]] = 0
        if not self.dijkstra(out_map, START_COORD):
            print('Map is invalid: No possible path found')
            out_map = self.generateMap()
        print('Valid path found')
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
        return True if END_COORD in distances.keys() else False


if __name__ == "__main__":
    env = Env()
    env.mainloop()
