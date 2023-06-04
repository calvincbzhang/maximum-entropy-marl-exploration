from __future__ import annotations

import numpy as np

from marl_grid.grid import Grid
from marl_grid.world_object import Wall
from marl_grid.marlgrid_env import MARLGridEnv
from marl_grid.actions import MiniActions

import random


class RoomsEnv(MARLGridEnv):

    def __init__(
        self, size, obstacle_type=Wall, max_steps: int | None = None, num_agents=1, **kwargs
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

        room_w = width // 2
        room_h = height // 2

        # For each row of rooms
        for j in range(0, 2):

            # For each column
            for i in range(0, 2):
                xL = i * room_w
                yT = j * room_h
                xR = xL + room_w
                yB = yT + room_h

                # Bottom wall and door
                if i + 1 < 2:
                    self.grid.vert_wall(xR, yT, room_h)
                    pos = (xR, self._rand_int(yT + 1, yB))
                    self.grid.set(*pos, None)

                # Bottom wall and door
                if j + 1 < 2:
                    self.grid.horz_wall(xL, yB, room_w)
                    pos = (self._rand_int(xL + 1, xR), yB)
                    self.grid.set(*pos, None)

        # Place the agents
        for a in range(self.num_agents):
            self.agent_pos[a] = self.place_agent(a)

    
class RoomsEnv10x10x2(RoomsEnv):
    def __init__(self):
        super().__init__(
            size=10,
            num_agents=2,
        )