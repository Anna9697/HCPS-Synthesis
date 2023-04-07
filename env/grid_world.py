import random
import time
import numpy as np
import tkinter as tk
from PIL import ImageTk, Image

np.random.seed(1)
PhotoImage = ImageTk.PhotoImage
UNIT = 100  # pixels
HEIGHT = 5  # grid height
WIDTH = 4  # grid width


class Env(tk.Tk):
    def __init__(self):
        super(Env, self).__init__()
        self.action_space = ['u', 'd', 'l', 'r']
        self.n_actions = len(self.action_space)
        self.title('monte carlo')
        self.geometry('{0}x{1}'.format(HEIGHT * UNIT, HEIGHT * UNIT))
        self.shapes = self.load_images()
        self.canvas = self._build_canvas()
        self.texts = []
        # self.absorb = []

    def _build_canvas(self):
        canvas = tk.Canvas(self, bg='white',
                                height=HEIGHT * UNIT,
                                width=WIDTH * UNIT)
        # create grids
        for c in range(0, WIDTH * UNIT, UNIT):  # 0~400 by 80
            x0, y0, x1, y1 = c, 0, c, HEIGHT * UNIT
            canvas.create_line(x0, y0, x1, y1)
        for r in range(0, HEIGHT * UNIT, UNIT):  # 0~400 by 80
            x0, y0, x1, y1 = 0, r, HEIGHT * UNIT, r
            canvas.create_line(x0, y0, x1, y1)

        # add img to canvas
        self.rectangle = canvas.create_image(0 * UNIT + UNIT/2, 0 * UNIT + UNIT/2, image=self.shapes[0])
        self.triangle1 = canvas.create_image(2 * UNIT + UNIT/2, 0 * UNIT + UNIT/2, image=self.shapes[1])
        self.triangle2 = canvas.create_image(2 * UNIT + UNIT/2, 2 * UNIT + UNIT/2, image=self.shapes[1])
        self.triangle3 = canvas.create_image(1 * UNIT + UNIT / 2, 4 * UNIT + UNIT / 2, image=self.shapes[1])
        self.triangle4 = canvas.create_image(2 * UNIT + UNIT / 2, 4 * UNIT + UNIT / 2, image=self.shapes[1])
        self.circle1 = canvas.create_image(2 * UNIT + UNIT/2, 1 * UNIT + UNIT/2, image=self.shapes[2])
        self.circle2 = canvas.create_image(2 * UNIT + UNIT / 2, 3 * UNIT + UNIT / 2, image=self.shapes[2])
        self.circle3 = canvas.create_image(0 * UNIT + UNIT / 2, 3 * UNIT + UNIT / 2, image=self.shapes[2])
        self.absracle = canvas.create_image(0 * UNIT + UNIT / 2, 2 * UNIT + UNIT / 2, image=self.shapes[3])

        # self.absorb.extend([self.triangle1, self.triangle2, self.triangle3, self.triangle4, self.circle2, self.circle3])
        # pack all
        canvas.pack()

        return canvas

    def load_images(self):
        rectangle = PhotoImage(
            Image.open("./images/rectangle.png").resize((65, 65)))
        triangle = PhotoImage(
            Image.open("./images/triangle.png").resize((65, 65)))
        circle = PhotoImage(
            Image.open("./images/circle.png").resize((65, 65)))
        abstracle = PhotoImage(
            Image.open("./images/abstracle.png").resize((100, 100)))

        return rectangle, triangle, circle, abstracle

    @staticmethod
    def coords_to_state(coords):
        x = int((coords[0] - 50) / 100)
        y = int((coords[1] - 50) / 100)
        return [x, y]

    def reset(self):
        self.update()
        # time.sleep(0.5)
        x, y = self.canvas.coords(self.rectangle)
        # x0, y0 = random.choice([[0, 0], [3, 0], [0, 4], [3, 4]])
        x0, y0 = [0, 0]
        mx = x0 * UNIT + UNIT / 2
        my = y0 * UNIT + UNIT / 2
        # self.canvas.move(self.rectangle, x * UNIT + UNIT / 2, y * UNIT + UNIT / 2)
        self.canvas.move(self.rectangle, mx - x, my - y)
        # return observation
        return self.coords_to_state(self.canvas.coords(self.rectangle))

    def step(self, action):
        state = self.canvas.coords(self.rectangle)
        base_action = np.array([0, 0])
        p = np.array([0.8, 0.1, 0.1])
        reward = 0
        # self.render()

        if action == 0: #up
            action = np.random.choice([0, 2, 3], p=p.ravel())
        if action == 1: #down
            action = np.random.choice([1, 2, 3], p=p.ravel())
        if action == 2: #left
            action = np.random.choice([2, 0, 1], p=p.ravel())
        if action == 3: #right
            action = np.random.choice([3, 0, 1], p=p.ravel())

        if action == 0:  # up
            if state[1] > UNIT:
                base_action[1] -= UNIT
        elif action == 1:  # down
            if state[1] < (HEIGHT - 1) * UNIT:
                base_action[1] += UNIT
        elif action == 2:  # left
            if state[0] > UNIT:
                base_action[0] -= UNIT
        elif action == 3:  # right
            if state[0] < (WIDTH - 1) * UNIT:
                base_action[0] += UNIT
        # move agent

        # if base_action != map(int, self.canvas.coords(self.absracle)):
        if state not in [self.canvas.coords(self.triangle1),
                                                      self.canvas.coords(self.triangle2),
                                                      self.canvas.coords(self.triangle3),
                                                      self.canvas.coords(self.triangle4),
                                                      self.canvas.coords(self.circle2),
                                                      self.canvas.coords(self.circle3)]:
            self.canvas.move(self.rectangle, base_action[0], base_action[1])
        if self.canvas.coords(self.rectangle) == self.canvas.coords(self.absracle):
            self.canvas.move(self.rectangle, -base_action[0], -base_action[1])
        # print(base_action)
        # move rectangle to top level of canvas
        self.canvas.tag_raise(self.rectangle)

        next_state = self.canvas.coords(self.rectangle)


        # reward function
        if next_state in [self.canvas.coords(self.circle1),
                          self.canvas.coords(self.circle2),
                          self.canvas.coords(self.circle3)]:
        # if next_state == self.canvas.coords(self.circle1) or next_state == self.canvas.coords(self.circle2) or next_state == self.canvas.coords(self.circle3):
            reward += 0.001
            # done = True
        elif next_state in [self.canvas.coords(self.triangle1),
                            self.canvas.coords(self.triangle2),
                            self.canvas.coords(self.triangle3),
                            self.canvas.coords(self.triangle4)]:
            reward -= 0.001
            # done = True
        else:
            reward += 0
            # done = False
        if next_state in [self.canvas.coords(self.triangle1),
                          self.canvas.coords(self.triangle2),
                          self.canvas.coords(self.triangle3),
                          self.canvas.coords(self.triangle4),
                          self.canvas.coords(self.circle2),
                          self.canvas.coords(self.circle3)]:
            # if next_state in [self.canvas.coords(self.circle2),
            #                   self.canvas.coords(self.circle3)]:
                # reward += 0.001
            done = True
        else:
            done = False

        next_state = self.coords_to_state(next_state)

        return next_state, reward, done

    def get_label(self, state):
        if state in [self.coords_to_state(self.canvas.coords(self.triangle1)),
                     self.coords_to_state(self.canvas.coords(self.triangle2)),
                     self.coords_to_state(self.canvas.coords(self.triangle3)),
                     self.coords_to_state(self.canvas.coords(self.triangle4))]:
            label ='c'
        elif state in [self.coords_to_state(self.canvas.coords(self.circle2)),
                       self.coords_to_state(self.canvas.coords(self.circle3))]:
            label = 'a'
        elif state in [self.coords_to_state(self.canvas.coords(self.circle1))]:
            label = 'b'
        else:
            label = 'd'
        return label

    def render(self):
        time.sleep(0.03)
        self.update()