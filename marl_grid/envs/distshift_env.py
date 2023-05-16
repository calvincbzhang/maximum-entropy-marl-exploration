from __future__ import annotations

import numpy as np

from marl_grid.grid import Grid
from marl_grid.world_object import Lava
from marl_grid.marlgrid_env import MARLGridEnv
from marl_grid.actions import MiniActions


class DistShiftEnv(MARLGridEnv):

    def __init__(
        self, size, obstacle_type=Lava, max_steps: int | None = None, num_agents=1, strip2_row=2, **kwargs
    ):
        self.obstacle_type = obstacle_type
        self.size = size
        self.num_agents = num_agents
        self.strip2_row = strip2_row

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

        # Place the lava rows
        for i in range(self.width - 6):
            self.grid.set(3 + i, 1, Lava())
            self.grid.set(3 + i, self.strip2_row, Lava())

        # Place the agents
        for a in range(self.num_agents):
            self.agent_pos[a] = self.place_agent(a)

    
class DistShiftEnv10x10x2(DistShiftEnv):
    def __init__(self):
        super().__init__(
            size=10,
            num_agents=2,
        )