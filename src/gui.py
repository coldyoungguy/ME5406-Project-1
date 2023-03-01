import tkinter as tk
from tkinter import ttk
from ttkthemes import ThemedTk
from params import *
from environment import *

GREY = "#323232"
DARK_GREY = "#171717"


class GUI(ThemedTk, object):
    def __init__(self):
        super(GUI, self).__init__()
        self.title('Frozen Lake Environment')
        self.geometry('400x600')
        self.configure(bg=GREY)

        self.env = Env()
        self.canvas_size = 400
        self.cell_size = self.canvas_size // GRID_SIZE

        self.cellMap = self.env.generateMap()
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
        self.mapGen_hole = tk.Spinbox(self.mapGen_frame, from_=0, to=1, format="%1.2f", increment=0.01,
                                      textvariable=tk.StringVar(self).set("0.30"))
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
                                    width=self.canvas_size,
                                    height=self.canvas_size)
        self.map_widget.grid(row=0, column=0)

        self.log_label = tk.Label(self.bottom_frame, text="Log")
        # self.log = tk.T

        self.createEnv()
        for widget in self.main_frame.winfo_children():
            for child in widget.winfo_children():
                child.grid_configure(padx=5, pady=5)
                child.configure(fg="white")

        print('Environment Initialised')

    def createEnv(self):
        print(f'Creating {GRID_SIZE}x{GRID_SIZE} grid')
        for r in range(GRID_SIZE):
            for c in range(GRID_SIZE):
                colour = DARK_GREY if self.cellMap[r][c] == 0 else 'white'
                self.drawRec(r, c, colour)

    def drawRec(self, r, c, colour):
        x1 = c * self.cell_size
        y1 = r * self.cell_size
        x2 = x1 + self.cell_size
        y2 = y1 + self.cell_size
        self.map_widget.create_rectangle(x1, y1, x2, y2, fill=colour)


if __name__ == "__main__":
    gui = GUI()
    gui.mainloop()
