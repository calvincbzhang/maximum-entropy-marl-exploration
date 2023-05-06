from __future__ import annotations

import hashlib
import math
from abc import abstractmethod
from typing import Any, Iterable, SupportsFloat, TypeVar

import gymnasium as gym
import numpy as np
import pygame
import pygame.freetype
from gymnasium import spaces
from gymnasium.core import ActType, ObsType

from marl_grid.actions import Actions
from marl_grid.constants import COLOR_NAMES, TILE_PIXELS
from marl_grid.grid import Grid
from marl_grid.world_object import Point, WorldObj, Agent

T = TypeVar("T")


class MARLGridEnv(gym.Env):
    """
    2D grid world game environment
    """

    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 10,
    }

    def __init__(
        self,
        grid_size: int | None = None,
        width: int | None = None,
        height: int | None = None,
        max_steps: int = 100,
        num_agents: int = 1,
        see_through_walls: bool = False,
        agent_view_size: int = 7,
        render_mode: str | None = None,
        screen_size: int | None = 640,
        highlight: bool = True,
        tile_size: int = TILE_PIXELS,
        actions = Actions,
    ):
        
        self.agents = range(num_agents)
        self.agent_pos = [None] * num_agents

        # Can't set both grid_size and width/height
        if grid_size:
            assert width is None and height is None
            width = grid_size
            height = grid_size
        assert width is not None and height is not None
        assert width >= 2 * num_agents and height >= 2 * num_agents

        # Action enumeration for this environment
        self.actions = actions

        # Actions are discrete integer values
        self.action_space = spaces.Discrete(len(self.actions))

        # Number of cells (width and height) in the agent view
        assert agent_view_size % 2 == 1
        assert agent_view_size >= 3
        self.agent_view_size = agent_view_size

        sample_obs = []
        for i in range(len(self.agents)):
            sample_obs.append([width, height])
        self.observation_space = spaces.MultiDiscrete(
            nvec=(sample_obs),
            dtype="uint8",
        )

        # Range of possible rewards
        self.reward_range = (0, 1)

        self.screen_size = screen_size
        self.render_size = None
        self.window = None
        self.clock = None

        # Environment configuration
        self.width = width
        self.height = height

        assert isinstance(
            max_steps, int
        ), f"The argument max_steps must be an integer, got: {type(max_steps)}"
        self.max_steps = max_steps

        self.see_through_walls = see_through_walls

        # Current grid and mission and carrying
        self.grid = Grid(width, height)
        self.carrying = [None] * num_agents

        # Rendering attributes
        self.render_mode = render_mode
        self.highlight = highlight
        self.tile_size = tile_size
        # self.agent_pov = agent_pov

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[ObsType, dict[str, Any]]:
        super().reset(seed=seed)

        # Generate a new random grid at the start of each episode
        self._gen_grid(self.width, self.height)

        obs = np.empty((len(self.agents), 2), dtype=np.uint8)

        # These fields should be defined by _gen_grid
        for a in self.agents:
            assert self.agent_pos[a] is not None
            obs[a] = self.agent_pos[a]

        # Item picked up, being carried, initially nothing
        for a in self.agents:
            self.carrying[a] = None

        # Step count since episode start
        self.step_count = 0

        if self.render_mode == "human":
            self.render()

        # Return first observation
        # obs = self.gen_obs()

        return obs, {}

    def hash(self, size=16):
        """Compute a hash that uniquely identifies the current state of the environment.
        :param size: Size of the hashing
        """
        sample_hash = hashlib.sha256()

        to_encode = [self.grid.encode().tolist(), self.agent_pos, self.agent_dir]
        for item in to_encode:
            sample_hash.update(str(item).encode("utf8"))

        return sample_hash.hexdigest()[:size]

    @property
    def steps_remaining(self):
        return self.max_steps - self.step_count

    # def __str__(self):
    #     """
    #     Produce a pretty string of the environment's grid along with the agent.
    #     A grid cell is represented by 2-character string, the first one for
    #     the object and the second one for the color.
    #     """

    #     # Map of object types to short string
    #     OBJECT_TO_STR = {
    #         "wall": "W",
    #         "floor": "F",
    #         "door": "D",
    #         "key": "K",
    #         "ball": "A",
    #         "box": "B",
    #         "goal": "G",
    #         "lava": "V",
    #     }

    #     # Map agent's direction to short string
    #     AGENT_DIR_TO_STR = {0: ">", 1: "V", 2: "<", 3: "^"}

    #     output = ""

    #     for j in range(self.grid.height):
    #         for i in range(self.grid.width):
    #             if i == self.agent_pos[0] and j == self.agent_pos[1]:
    #                 output += 2 * AGENT_DIR_TO_STR[self.agent_dir]
    #                 continue

    #             tile = self.grid.get(i, j)

    #             if tile is None:
    #                 output += "  "
    #                 continue

    #             if tile.type == "door":
    #                 if tile.is_open:
    #                     output += "__"
    #                 elif tile.is_locked:
    #                     output += "L" + tile.color[0].upper()
    #                 else:
    #                     output += "D" + tile.color[0].upper()
    #                 continue

    #             output += OBJECT_TO_STR[tile.type] + tile.color[0].upper()

    #         if j < self.grid.height - 1:
    #             output += "\n"

    #     return output

    @abstractmethod
    def _gen_grid(self, width, height):
        pass

    def _reward(self) -> float:
        """
        Compute the reward to be given upon success
        """

        return 1 - 0.9 * (self.step_count / self.max_steps)

    def _rand_int(self, low: int, high: int) -> int:
        """
        Generate random integer in [low,high[
        """

        return self.np_random.integers(low, high)

    def _rand_float(self, low: float, high: float) -> float:
        """
        Generate random float in [low,high[
        """

        return self.np_random.uniform(low, high)

    def _rand_bool(self) -> bool:
        """
        Generate random boolean value
        """

        return self.np_random.integers(0, 2) == 0

    def _rand_elem(self, iterable: Iterable[T]) -> T:
        """
        Pick a random element in a list
        """

        lst = list(iterable)
        idx = self._rand_int(0, len(lst))
        return lst[idx]

    def _rand_subset(self, iterable: Iterable[T], num_elems: int) -> list[T]:
        """
        Sample a random subset of distinct elements of a list
        """

        lst = list(iterable)
        assert num_elems <= len(lst)

        out: list[T] = []

        while len(out) < num_elems:
            elem = self._rand_elem(lst)
            lst.remove(elem)
            out.append(elem)

        return out

    def _rand_color(self) -> str:
        """
        Generate a random color name (string)
        """

        return self._rand_elem(COLOR_NAMES)

    def _rand_pos(
        self, x_low: int, x_high: int, y_low: int, y_high: int
    ) -> tuple[int, int]:
        """
        Generate a random (x,y) position tuple
        """

        return (
            self.np_random.integers(x_low, x_high),
            self.np_random.integers(y_low, y_high),
        )

    def place_obj(
        self,
        obj: WorldObj | None,
        top: Point = None,
        size: tuple[int, int] = None,
        reject_fn=None,
        max_tries=math.inf,
    ):
        """
        Place an object at an empty position in the grid
        :param top: top-left position of the rectangle where to place
        :param size: size of the rectangle where to place
        :param reject_fn: function to filter out potential positions
        """

        if top is None:
            top = (0, 0)
        else:
            top = (max(top[0], 0), max(top[1], 0))

        if size is None:
            size = (self.grid.width, self.grid.height)

        num_tries = 0

        while True:
            # This is to handle with rare cases where rejection sampling
            # gets stuck in an infinite loop
            if num_tries > max_tries:
                raise RecursionError("rejection sampling failed in place_obj")

            num_tries += 1

            pos = (
                self._rand_int(top[0], min(top[0] + size[0], self.grid.width)),
                self._rand_int(top[1], min(top[1] + size[1], self.grid.height)),
            )

            # Don't place the object on top of another object
            if self.grid.get(*pos) is not None:
                continue

            # Don't place the object where the agent is
            if np.array_equal(pos, self.agent_pos):
                continue

            # Check if there is a filtering criterion
            if reject_fn and reject_fn(self, pos):
                continue

            break

        self.grid.set(pos[0], pos[1], obj)

        if obj is not None:
            obj.init_pos = pos
            obj.cur_pos = pos

        return pos

    def put_obj(self, obj: WorldObj, i: int, j: int):
        """
        Put an object at a specific position in the grid
        """

        self.grid.set(i, j, obj)
        obj.init_pos = (i, j)
        obj.cur_pos = (i, j)

    def place_agent(self, agent, top=None, size=None, max_tries=math.inf):
        """
        Set the agent's starting point at an empty position in the grid
        """

        self.agent_pos[agent] = None
        pos = self.place_obj(Agent(), top, size, max_tries=max_tries)
        self.agent_pos[agent] = pos

        return pos

    def step(
        self, action
    ) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        self.step_count += 1

        order = np.random.permutation(len(action))

        reward = np.zeros(len(action))
        terminated = False
        truncated = False

        obs = np.empty((len(action), 2), dtype=np.uint8)

        for i in order:

            # Get the position in the cell the agent will be after executing the action
            fwd_pos = self.agent_pos[i] \
                + (action[i] == self.actions.up) * np.array([0, -1]) \
                + (action[i] == self.actions.down) * np.array([0, 1]) \
                + (action[i] == self.actions.left) * np.array([-1, 0]) \
                + (action[i] == self.actions.right) * np.array([1, 0])

            # Get the contents of the cell in front of the agent
            fwd_cell = self.grid.get(*fwd_pos)

            obs[i] = fwd_pos

            if action[i] == self.actions.nothing:
                pass

            elif action[i] in range(1,5):
                if fwd_cell is None or fwd_cell.can_overlap():
                    self.grid.set(self.agent_pos[i][0], self.agent_pos[i][1], None)
                    self.agent_pos[i] = tuple(fwd_pos)
                    self.grid.set(self.agent_pos[i][0], self.agent_pos[i][1], Agent())
                if fwd_cell is not None and fwd_cell.type == "goal":
                    terminated = True
                    reward[i] = self._reward()
                if fwd_cell is not None and (fwd_cell.type == "lava" or fwd_cell.type == "agent"):
                    # if fwd_cell.type == "agent":
                    #     print("Agent collision")
                    # elif fwd_cell.type == "lava":
                    #     print("Lava collision")
                    terminated = True

            elif len(self.actions) > 5:
                # Pick up an object
                if action[i] == self.actions.pickup:
                    if fwd_cell and fwd_cell.can_pickup():
                        if self.carrying[i] is None:
                            self.carrying[i] = fwd_cell
                            self.carrying[i].cur_pos = np.array([-1, -1])
                            self.grid.set(fwd_pos[0], fwd_pos[1], None)

                # Drop an object
                elif action[i] == self.actions.drop:
                    if not fwd_cell and self.carrying:
                        self.grid.set(fwd_pos[0], fwd_pos[1], self.carrying[i])
                        self.carrying[i].cur_pos = fwd_pos
                        self.carrying[i] = None

                # Toggle/activate an object
                elif action[i] == self.actions.toggle:
                    if fwd_cell:
                        fwd_cell.toggle(self, fwd_pos)

                # Done action (not used by default)
                elif action[i] == self.actions.done:
                    pass

            else:
                raise ValueError(f"Unknown action: {action[i]}")

        if self.step_count >= self.max_steps:
            truncated = True

        if self.render_mode == "human":
            self.render()

        # obs = self.gen_obs()

        reward = np.mean(reward)

        return obs, reward, terminated, truncated, {}

    def gen_obs(self):
        """
        Generate the agent's view (fully observable)
        """
        obs = []
        # Encode the view into a numpy array
        return self.grid.encode()

    def get_full_render(self, highlight, tile_size):
        """
        Render a non-paratial observation for visualization
        """

        # Render the whole grid
        img = self.grid.render(
            tile_size,
            self.agent_pos,
        )

        return img

    def get_frame(
        self,
        highlight: bool = True,
        tile_size: int = TILE_PIXELS,
    ):
        """Returns an RGB image corresponding to the whole environment or the agent's point of view.
        Args:
            highlight (bool): If true, the agent's field of view or point of view is highlighted with a lighter gray color.
            tile_size (int): How many pixels will form a tile from the NxM grid.
            agent_pov (bool): If true, the rendered frame will only contain the point of view of the agent.
        Returns:
            frame (np.ndarray): A frame of type numpy.ndarray with shape (x, y, 3) representing RGB values for the x-by-y pixel image.
        """
        return self.get_full_render(highlight, tile_size)

    def render(self):
        img = self.get_frame(self.highlight, self.tile_size)

        if self.render_mode == "human":
            img = np.transpose(img, axes=(1, 0, 2))
            if self.render_size is None:
                self.render_size = img.shape[:2]
            if self.window is None:
                pygame.init()
                pygame.display.init()
                self.window = pygame.display.set_mode(
                    (self.screen_size, self.screen_size)
                )
                pygame.display.set_caption("minigrid")
            if self.clock is None:
                self.clock = pygame.time.Clock()
            surf = pygame.surfarray.make_surface(img)

            # Create background with mission description
            offset = surf.get_size()[0] * 0
            # offset = 32 if self.agent_pov else 64
            bg = pygame.Surface(
                (int(surf.get_size()[0] + offset), int(surf.get_size()[1] + offset))
            )
            bg.convert()
            bg.fill((255, 255, 255))
            bg.blit(surf, (offset / 2, 0))

            bg = pygame.transform.smoothscale(bg, (self.screen_size, self.screen_size))

            self.window.blit(bg, (0, 0))
            pygame.event.pump()
            self.clock.tick(self.metadata["render_fps"])
            pygame.display.flip()

        elif self.render_mode == "rgb_array":
            return img

    def close(self):
        if self.window:
            pygame.quit()