import tkinter as tk
import numpy as np
import pandas as pd
from environment import *

grid_size = 10
cell_size = 50
cellMap = np.random.randint(2, size=(grid_size, grid_size))


def create():
    for row in range(grid_size):
        for col in range(grid_size):
            color = 'white' if cellMap[row][col] == 1 else 'black'
            draw(row, col, color)


def draw(row, col, color):
    x1 = col * cell_size
    y1 = row * cell_size
    x2 = x1 + cell_size
    y2 = y1 + cell_size
    ffs.create_rectangle(x1, y1, x2, y2, fill=color)


window = tk.Tk()
canvas_side = grid_size * cell_size
ffs = tk.Canvas(window, width=canvas_side, height=canvas_side, bg='grey')
# ffs.pack()

create()
ffs.pack()
# window.mainloop()

# d = {}
# for r in range(4):
#     for c in range(4):
#         d[(r, c)] = [0] * 4
#
# print(d)
#
# from params import *
# print(END_COORD)
env = Env()
t = pd.DataFrame(columns=env.action_space)
print(t)
print((2,1) in t)
t = t.append((2,1))
print((2,1) in t)
path = ((0,1), (0,2), (1,0), (0,1))

def add_state_to_table(state, table):
    if state not in table:
        return table.append(pd.Series([0] * 4,
                    index=table.columns, name=state))
    else:
        return table

for i in path:
    t = add_state_to_table(i, t)

print(t)