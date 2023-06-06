from __future__ import annotations

import numpy as np

from marl_grid.grid import Grid
from marl_grid.world_object import Wall
from marl_grid.marlgrid_env import MARLGridEnv
from marl_grid.actions import MiniActions


class MazeEnv(MARLGridEnv):

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

        walls = [[1,4], [2,1], [2,2], [2,4], [2,6], [2,7], [2,8], [2,9], [3,4],
                 [3,6], [3,9], [4,2], [4,3], [4,4], [4,6], [4,9], [5,6], [6,1],
                 [6,2], [6,3], [6,4], [6,5], [6,6], [6,7], [6,8], [6,9], [6,10],
                 [7,3], [7,7], [7,8], [7,9], [7,10], [8,1], [8,3], [8,5], [8,7],
                 [9,1], [9,3], [9,5], [9,7], [9,9], [10,1], [10,5], [10,9]]
        
        for w in walls:
            self.grid.set(w[0], w[1], Wall())

        # Place the agents
        for a in range(self.num_agents):
            self.agent_pos[a] = self.place_agent(a)

    
class MazeEnv10x10x2(MazeEnv):
    def __init__(self):
        super().__init__(
            size=10,
            num_agents=2,
        )