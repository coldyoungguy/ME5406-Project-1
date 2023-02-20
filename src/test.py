import tkinter as tk
import numpy as np

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
window.mainloop()
