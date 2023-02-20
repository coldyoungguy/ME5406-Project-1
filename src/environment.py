import heapq
import tkinter as tk
import numpy as np
from params import *


class Env(tk.Tk, object):
    def __init__(self):
        super(Env, self).__init__()
        self.title('Frozen Lake')
        self.grid_size = GRID_SIZE  # 10
        self.cell_size = CELL_SIZE
        self.action_space = ['up', 'down', 'left', 'right']
        self.num_states = len(self.action_space)
        self.num_states = self.grid_size ** 2

        self.cellMap = self.generateMap()
        self.widgets = tk.Canvas(self, bg='grey',
                                 width=self.grid_size * self.cell_size,
                                 height=self.grid_size * self.cell_size)
        self.widgets.pack()
        self.createEnv()
        print('Environment Initialised')

    def createEnv(self):
        print(f'Creating {self.grid_size}x{self.grid_size} grid')
        for r in range(self.grid_size):
            for c in range(self.grid_size):
                colour = 'white' if self.cellMap[r][c] == 0 else 'black'
                self.drawRec(r, c, colour)

    def drawRec(self, r, c, colour):
        x1 = c * self.cell_size
        y1 = r * self.cell_size
        x2 = x1 + self.cell_size
        y2 = y1 + self.cell_size
        self.widgets.create_rectangle(x1, y1, x2, y2, fill=colour)

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
