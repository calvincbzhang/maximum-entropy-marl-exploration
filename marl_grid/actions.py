from __future__ import annotations

from enum import IntEnum


class Actions(IntEnum):
    # Go up, down, right, left or do nothing
    nothing = 0
    up = 1
    down = 2
    right = 3
    left = 4
    # Pick up a object
    pickup = 5
    # Drop an object
    drop = 6
    # Toggle/activate an object
    toggle = 7

    # Done completing task
    done = 8


class MiniActions(IntEnum):
    # Go up, down, right, left or do nothing
    nothing = 0
    up = 1
    down = 2
    right = 3
    left = 4
