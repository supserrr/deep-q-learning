"""Draws Pong in a minimal black-and-white style for play.py.

This only changes how the game is shown. The agent still sees the original Atari frames,
so the trained model works unchanged. Each frame, we find the paddles and the ball in the
Atari screen and redraw them as white shapes on black, with a dashed centre line and the score.
"""

import gymnasium as gym
import imageio.v2 as imageio
import numpy as np
import pygame

WIDTH, HEIGHT = 800, 400  # window size (2:1, like an arcade screen)
TOP, BOTTOM = 34, 194  # rows of the 210x160 Atari screen that hold the playing field
SCALE_X, SCALE_Y = WIDTH / 160, HEIGHT / (BOTTOM - TOP)

BLACK, WHITE, GREY = (0, 0, 0), (255, 255, 255), (140, 140, 140)

# Colours in the Atari screen, and the columns where each paddle moves
COMPUTER = (213, 130, 74), (16, 20)  # left paddle (orange)
AGENT = (92, 186, 92), (140, 144)  # right paddle (green): our DQN agent
BALL = (236, 236, 236), (0, 160)

# Blocky 3x5 digits for the score
DIGITS = {
    "0": ["###", "#.#", "#.#", "#.#", "###"], "1": ["..#", "..#", "..#", "..#", "..#"],
    "2": ["###", "..#", "###", "#..", "###"], "3": ["###", "..#", "###", "..#", "###"],
    "4": ["#.#", "#.#", "###", "..#", "..#"], "5": ["###", "#..", "###", "..#", "###"],
    "6": ["###", "#..", "###", "#.#", "###"], "7": ["###", "..#", "..#", "..#", "..#"],
    "8": ["###", "#.#", "###", "#.#", "###"], "9": ["###", "#.#", "###", "..#", "###"],
}
CELL = 12  # size of one block of a digit, in pixels


def find(frame, thing):
    """Bounding box (x0, y0, x1, y1) of one object in the playing field, in Atari pixels, or None."""
    color, (col0, col1) = thing
    field = frame[TOP:BOTTOM, col0:col1]
    ys, xs = np.nonzero(np.all(field == color, axis=-1))
    if len(ys) == 0:
        return None
    return xs.min() + col0, ys.min(), xs.max() + col0 + 1, ys.max() + 1


class RetroView(gym.Wrapper):
    """Wraps the raw Atari game and redraws every frame (60 per second) in its own window.

    With video_folder set, frames are saved to an .mp4 per game instead of shown in a window.
    """

    def __init__(self, env, video_folder=None):
        super().__init__(env)
        pygame.init()
        self.video_folder = video_folder
        self.screen = pygame.Surface((WIDTH, HEIGHT)) if video_folder else pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("DQN agent playing Pong")
        self.clock = pygame.time.Clock()
        self.score = [0, 0]  # computer, agent
        self.games = 0
        self.writer = None

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.score = [0, 0]
        if self.video_folder:
            if self.writer:
                self.writer.close()
            self.writer = imageio.get_writer(f"{self.video_folder}/pong-dqn-episode-{self.games}.mp4", fps=60)
        self.games += 1
        self.draw(obs)
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        if reward > 0:
            self.score[1] += 1
        elif reward < 0:
            self.score[0] += 1
        self.draw(obs)
        return obs, reward, terminated, truncated, info

    def draw(self, frame):
        self.screen.fill(BLACK)
        for y in range(0, HEIGHT, 24):  # dashed centre line
            pygame.draw.rect(self.screen, GREY, (WIDTH // 2 - 5, y + 6, 10, 12))
        self.draw_number(self.score[0], right_edge=WIDTH // 2 - 32)
        self.draw_number(self.score[1], left_edge=WIDTH // 2 + 32)

        for paddle in (COMPUTER, AGENT):
            box = find(frame, paddle)
            if box:
                x0, y0, x1, y1 = box
                center_x = (x0 + x1) / 2 * SCALE_X
                pygame.draw.rect(self.screen, WHITE, (center_x - 6, y0 * SCALE_Y, 12, (y1 - y0) * SCALE_Y))
        ball = find(frame, BALL)
        if ball:
            x0, y0, x1, y1 = ball
            pygame.draw.rect(self.screen, WHITE, ((x0 + x1) / 2 * SCALE_X - 6, (y0 + y1) / 2 * SCALE_Y - 6, 12, 12))

        if self.writer:
            self.writer.append_data(pygame.surfarray.array3d(self.screen).swapaxes(0, 1))
        else:
            pygame.display.flip()
            self.clock.tick(60)  # real time: the Atari runs at 60 frames per second
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    raise SystemExit

    def draw_number(self, number, left_edge=None, right_edge=None):
        """Draw a score in blocky digits, either starting at left_edge or ending at right_edge."""
        text = str(number)
        width = len(text) * 4 * CELL - CELL
        x = left_edge if left_edge is not None else right_edge - width
        for char in text:
            for row, line in enumerate(DIGITS[char]):
                for col, block in enumerate(line):
                    if block == "#":
                        pygame.draw.rect(self.screen, GREY, (x + col * CELL, 16 + row * CELL, CELL, CELL))
            x += 4 * CELL

    def close(self):
        if self.writer:
            self.writer.close()
        pygame.quit()
        super().close()
