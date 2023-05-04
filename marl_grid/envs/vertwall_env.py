from __future__ import annotations

import numpy as np

from marl_grid.grid import Grid
from marl_grid.world_object import Lava
from marl_grid.marlgrid_env import MARLGridEnv
from marl_grid.actions import MiniActions


class VertWallEnv(MARLGridEnv):

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

        # Place the agents
        for a in range(self.num_agents):
            self.agent_pos[a] = self.place_agent(a)

        # Generate and store random gap position
        self.gap_pos = np.array(
            (
                self._rand_int(2, width - 2),
                self._rand_int(1, height - 1),
            )
        )

        # Place the obstacle wall
        self.grid.vert_wall(self.gap_pos[0], 1, height - 2, self.obstacle_type)

        # Put a hole in the wall
        self.grid.set(*self.gap_pos, None)

    
class VertWallEnv10x10x2(VertWallEnv):
    def __init__(self):
        super().__init__(
            size=10,
            num_agents=2,
        )