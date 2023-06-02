from __future__ import annotations

import numpy as np

from marl_grid.grid import Grid
from marl_grid.world_object import Lava
from marl_grid.marlgrid_env import MARLGridEnv
from marl_grid.actions import MiniActions

import random


class RoomsEnv(MARLGridEnv):

    def __init__(
        self, size, obstacle_type=Lava, max_steps: int | None = None, num_agents=1, **kwargs
    ):
        self.obstacle_type = obstacle_type
        self.size = size
        self.num_agents = num_agents

        if max_steps is None:
            max_steps = 4 * size**2

        super().__init__(
            width=size,
            height=size,
            num_agents=num_agents,
            see_through_walls=False,
            max_steps=max_steps,
            render_mode='human',
            actions = MiniActions,
            **kwargs,
        )

    def _gen_grid(self, width, height):
        assert width >= 5 and height >= 5, "The grid should be at least 5x5."

        # Create an empty grid
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)

        roomW = width // 3
        roomH = height // 3

        # For each row of rooms
        for j in range(0, 3):

            # For each column
            for i in range(0, 3):
                xL = i * roomW
                yT = j * roomH
                xR = xL + roomW
                yB = yT + roomH

                # Bottom wall and door
                if i + 1 < 3:
                    self.grid.vert_wall(xR, yT, roomH)
                    pos = (xR, self._rand_int(yT + 1, yB - 1))
                    self.grid.set(*pos, None)

                # Bottom wall and door
                if j + 1 < 3:
                    self.grid.horz_wall(xL, yB, roomW)
                    pos = (self._rand_int(xL + 1, xR - 1), yB)
                    self.grid.set(*pos, None)

        # Place the lava walls
        for i in range(1, width-1, 2):
            for j in range(1, height-1, 2):
                if random.random() < 0.2:  # Adjust the lava percentage as desired
                    self.grid.set(i, j, Lava())
                    self.grid.set(i, j, Lava())

        # Place the agents
        for a in range(self.num_agents):
            self.agent_pos[a] = self.place_agent(a)

    
class RoomsEnv10x10x2(RoomsEnv):
    def __init__(self):
        super().__init__(
            size=10,
            num_agents=2,
        )