from __future__ import annotations

import numpy as np

from marl_grid.grid import Grid
from marl_grid.world_object import Wall
from marl_grid.marlgrid_env import MARLGridEnv
from marl_grid.actions import MiniActions

import random


class RoomsEnv(MARLGridEnv):

    def __init__(
        self, height, width, obstacle_type=Wall, max_steps: int | None = None, num_agents=1, initial_positions=None, **kwargs
    ):
        self.obstacle_type = obstacle_type
        self.num_agents = num_agents

        if max_steps is None:
            max_steps = 4 * int(np.maximum(width, height))**2

        super().__init__(
            width=width,
            height=height,
            initial_positions=initial_positions,
            num_agents=num_agents,
            see_through_walls=False,
            max_steps=max_steps,
            render_mode='human',
            actions = MiniActions,
            **kwargs,
        )

    def _gen_grid(self, width, height):
        # assert width >= 5 and height >= 5, "The grid should be at least 5x5."

        # Create an empty grid
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)

        walls = [
            [5, 1], [5, 3], [5, 4], [5, 5], [5, 6], [5, 7], [5, 9],
            [1, 5], [3, 5], [4, 5], [6, 5], [7, 5], [9, 5]
        ]

        for w in walls:
            self.grid.set(w[0], w[1], Wall())

        # Place the agents
        for a in range(self.num_agents):
            self.agent_pos[a] = self.place_agent(a)

    
class RoomsEnv10x10x2(RoomsEnv):
    def __init__(self):
        super().__init__(
            size=10,
            num_agents=2,
        )