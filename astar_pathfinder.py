import heapq
from typing import Dict, List, Set, Tuple, Optional

from utils import manhattan_distance


class Node:
    def __init__(self, position: Tuple[int, int], g_cost: int = 0, h_cost: int = 0, parent: Optional['Node'] = None):
        self.position = position
        self.g_cost = g_cost  # Cost from start to current node
        self.h_cost = h_cost  # Estimated cost from current node to goal
        self.f_cost = g_cost + h_cost  # Total cost
        self.parent = parent

    def __lt__(self, other):
        return self.f_cost < other.f_cost


class BattlesnakePathfinder:
    def __init__(self, game_state: Dict):
        self.game_state = game_state
        self.board_width = game_state['board']['width']
        self.board_height = game_state['board']['height']
        self.directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # up, right, down, left

    def get_neighbors(self, position: Tuple[int, int], obstacles: Set[Tuple[int, int]]) -> List[Tuple[int, int]]:
        """Get valid neighboring positions."""
        neighbors = []
        x, y = position

        for dx, dy in self.directions:
            new_x, new_y = x + dx, y + dy

            # Check bounds and obstacles
            if (0 <= new_x < self.board_width and
                    0 <= new_y < self.board_height and
                    (new_x, new_y) not in obstacles):
                neighbors.append((new_x, new_y))

        return neighbors

    def find_path(self, start: Dict, goal: Dict) -> Optional[List[str]]:
        """Find path from start to goal using A* algorithm."""
        start_pos = (start['x'], start['y'])
        goal_pos = (goal['x'], goal['y'])

        # Create set of obstacles (snake bodies and hazards)
        obstacles = set()

        # Add all snake body segments to obstacles
        for snake in self.game_state['board']['snakes']:
            for segment in snake['body']:
                obstacles.add((segment['x'], segment['y']))

        # Add hazards to obstacles (if any)
        if 'hazards' in self.game_state['board']:
            for hazard in self.game_state['board']['hazards']:
                obstacles.add((hazard['x'], hazard['y']))

        # Initialize open and closed sets
        open_set = []
        closed_set = set()

        # Create start node
        start_node = Node(
            start_pos,
            g_cost=0,
            h_cost=manhattan_distance(start_pos, goal_pos)
        )

        # Add start node to open set
        heapq.heappush(open_set, start_node)

        while open_set:
            current = heapq.heappop(open_set)

            if current.position == goal_pos:
                # Path found, convert to directions
                return self._convert_path_to_directions(self._reconstruct_path(current))

            closed_set.add(current.position)

            # Check all neighbors
            for neighbor_pos in self.get_neighbors(current.position, obstacles):
                if neighbor_pos in closed_set:
                    continue

                g_cost = current.g_cost + 1
                h_cost = manhattan_distance(neighbor_pos, goal_pos)

                neighbor = Node(
                    neighbor_pos,
                    g_cost=g_cost,
                    h_cost=h_cost,
                    parent=current
                )
                heapq.heappush(open_set, neighbor)

        return None  # No path found

    def _reconstruct_path(self, end_node: Node) -> List[Tuple[int, int]]:
        """Reconstruct path from end node to start node."""
        path = []
        current = end_node

        while current is not None:
            path.append(current.position)
            current = current.parent

        return path[::-1]  # Reverse path to get start to end

    def _convert_path_to_directions(self, path: List[Tuple[int, int]]) -> List[str]:
        """Convert coordinate path to direction commands."""
        if len(path) < 2:
            return []

        directions = []
        for i in range(len(path) - 1):
            current = path[i]
            next_pos = path[i + 1]

            # Calculate direction
            dx = next_pos[0] - current[0]
            dy = next_pos[1] - current[1]

            if dx == 1:
                directions.append("right")
            elif dx == -1:
                directions.append("left")
            elif dy == 1:
                directions.append("up")
            elif dy == -1:
                directions.append("down")

        return directions
