from typing import Tuple


def manhattan_distance(start: Tuple[int, int], goal: Tuple[int, int]) -> int:
    """Calculate Manhattan distance between two points."""
    return abs(start[0] - goal[0]) + abs(start[1] - goal[1])
