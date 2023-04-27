from __future__ import annotations

import numpy as np

from marl_grid.grid import Grid
from marl_grid.world_object import Goal, Lava
from marl_grid.marlgrid_env import MARLGridEnv


class EmptyEnv(MARLGridEnv):

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
            **kwargs,
        )

    @staticmethod
    def _gen_mission_lava():
        return "avoid the lava and get to the green goal square"

    @staticmethod
    def _gen_mission():
        return "find the opening and get to the green goal square"

    def _gen_grid(self, width, height):
        assert width >= 5 and height >= 5

        # Create an empty grid
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)

        # Place the agent in the top-left corner
        for a in range(self.num_agents):
            self.agent_pos = np.array((a+1, a+1))

        # Place a goal square in the bottom-right corner
        self.goal_pos = np.array((width - 2, height - 2))
        self.put_obj(Goal(), *self.goal_pos)

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

        self.mission = (
            "avoid the lava and get to the green goal square"
            if self.obstacle_type == Lava
            else "find the opening and get to the green goal square"
        )

class EmptyEnvTest(EmptyEnv):
    def __init__(self):
        super().__init__(
            size=20,
            num_agents=5,
        )