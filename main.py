# Welcome to
# __________         __    __  .__                               __
# \______   \_____ _/  |__/  |_|  |   ____   ______ ____ _____  |  | __ ____
#  |    |  _/\__  \\   __\   __\  | _/ __ \ /  ___//    \\__  \ |  |/ // __ \
#  |    |   \ / __ \|  |  |  | |  |_\  ___/ \___ \|   |  \/ __ \|    <\  ___/
#  |________/(______/__|  |__| |____/\_____>______>___|__(______/__|__\\_____>
#
# This file can be a nice home for your Battlesnake logic and helper functions.
#
# To get you started we've included code to prevent your Battlesnake from moving backwards.
# For more info see docs.battlesnake.com


import random
from typing import Dict, List, Optional, Union

from astar_pathfinder import BattlesnakePathfinder
from utils import manhattan_distance
from collections import deque


# info is called when you create your Battlesnake on play.battlesnake.com
# and controls your Battlesnake's appearance
# TIP: If you open your Battlesnake URL in a browser you should see this data
def info() -> Dict:
    print("INFO")

    return {
        "apiversion": "1",
        "author": "cs5446-g37",
        "color": "#FAC3A4",
        "head": "smart-caterpillar",
        "tail": "mouse",
    }


# start is called when your Battlesnake begins a game
def start(game_state: Dict):
    print("GAME START")


# end is called when your Battlesnake finishes a game
def end(game_state: Dict):
    print("GAME OVER\n")


# move is called on every turn and returns your next move
# Valid moves are "up", "down", "left", or "right"
# See https://docs.battlesnake.com/api/example-move for available data
def move(game_state: Dict) -> Dict:
    is_move_safe = {"up": True, "down": True, "left": True, "right": True}

    # We've included code to prevent your Battlesnake from moving backwards
    my_head = game_state["you"]["body"][0]  # Coordinates of your head
    my_neck = game_state["you"]["body"][1]  # Coordinates of your "neck"

    if my_neck["x"] < my_head["x"]:  # Neck is left of head, don't move left
        is_move_safe["left"] = False

    elif my_neck["x"] > my_head["x"]:  # Neck is right of head, don't move right
        is_move_safe["right"] = False

    elif my_neck["y"] < my_head["y"]:  # Neck is below head, don't move down
        is_move_safe["down"] = False

    elif my_neck["y"] > my_head["y"]:  # Neck is above head, don't move up
        is_move_safe["up"] = False

    # Step 1 - Prevent your Battlesnake from moving out of bounds
    board_width = game_state['board']['width']
    board_height = game_state['board']['height']

    if my_head["x"] == 0:
        is_move_safe["left"] = False

    if my_head["x"] == board_width - 1:
        is_move_safe["right"] = False

    if my_head["y"] == 0:
        is_move_safe["down"] = False

    if my_head["y"] == board_height - 1:
        is_move_safe["up"] = False

    # Step 2 - Prevent your Battlesnake from colliding with itself
    my_body = game_state['you']['body']
    for segment in my_body[1:]:  # Skip the head
        # Check potential next positions against each body segment
        if {"x": my_head["x"], "y": my_head["y"] + 1} == segment:
            is_move_safe["up"] = False
        if {"x": my_head["x"], "y": my_head["y"] - 1} == segment:
            is_move_safe["down"] = False
        if {"x": my_head["x"] - 1, "y": my_head["y"]} == segment:
            is_move_safe["left"] = False
        if {"x": my_head["x"] + 1, "y": my_head["y"]} == segment:
            is_move_safe["right"] = False
    # Step 3 - Prevent your Battlesnake from colliding with other Battlesnakes
    opponents = game_state['board']['snakes']

    for snake in opponents:
        for segment in snake['body']:
            if {"x": my_head["x"], "y": my_head["y"] + 1} == segment:
                is_move_safe["up"] = False
            if {"x": my_head["x"], "y": my_head["y"] - 1} == segment:
                is_move_safe["down"] = False
            if {"x": my_head["x"] - 1, "y": my_head["y"]} == segment:
                is_move_safe["left"] = False
            if {"x": my_head["x"] + 1, "y": my_head["y"]} == segment:
                is_move_safe["right"] = False

    # Are there any safe moves left?
    safe_moves = []
    for move, isSafe in is_move_safe.items():
        if isSafe:
            safe_moves.append(move)
        else:
            print(f"UnSafe moves after initial checks: {move}")

    if len(safe_moves) == 0:
        return {"move": "down"}

   #step 4- move to food
    next_move = None
    food = game_state['board']['food']
    if len(food) > 0:
        # Find closest food
        target_food = get_closest_food(my_head, food)
        print(f"Target food at: {target_food}")

        # Initialize variables to track best move
        max_accessible_area = -1
        best_move = None

        # Get path to food using A* pathfinding
        pathfinder = BattlesnakePathfinder(game_state)
        path = pathfinder.find_path(my_head, target_food)
        print(f"Path found: {path}")

        if path and len(path) > 0:
            potential_next_move = path[0]

            # Calculate accessible area for the move suggested by pathfinding
            new_head = simulate_move(my_head, potential_next_move)
            path_accessible_area = flood_fill(new_head, game_state)

            # Calculate minimum safe area based on snake length
            min_area_threshold = len(game_state['you']['body']) * 1.25

            # Check if the pathfinding move leads to sufficient space
            if path_accessible_area >= min_area_threshold:
                max_accessible_area = path_accessible_area
                best_move = potential_next_move

            # If pathfinding move isn't safe enough, evaluate all safe moves
            if best_move is None:
                for move in safe_moves:
                    new_head = simulate_move(my_head, move)
                    area = flood_fill(new_head, game_state)

                    # Update best move if this move leads to larger accessible area
                    if area > max_accessible_area and area >= min_area_threshold:
                        max_accessible_area = area
                        best_move = move
        else:
            # No path found, use improved naive move that considers accessible area
            best_move = get_naive_move(my_head, target_food, safe_moves, game_state)

        # Final fallback if no move has been selected
        if best_move is None and safe_moves:
            next_move = choose_move_with_max_accessible_area(safe_moves, my_head, game_state)
        else:
            next_move = best_move

    else:
        # No food, move to position with maximum accessible area
        next_move = choose_move_with_max_accessible_area(safe_moves, my_head, game_state)

    # Final safety check
    if next_move is None and safe_moves:
        next_move = safe_moves[0]
    elif next_move is None:
        next_move = "up"
    return {"move": next_move}

def get_closest_food(head: Dict, food_list: List[Dict]) -> Optional[Dict]:
    """Find the closest food item using Manhattan distance."""
    if not food_list:
        return None

    closest_food = min(
        food_list,
        key=lambda food: manhattan_distance((head['x'], head['y']), (food['x'], food['y']))
    )
    return closest_food


# def get_naive_move(start_pos: Dict, goal_pos: Dict, safe_moves: List[str]) -> Union[str, None]:
#     if not safe_moves:
#         return None
#
#     # Try to move towards the goal if it's safe
#     if start_pos["x"] < goal_pos["x"] and "right" in safe_moves:
#         next_move = "right"
#     elif start_pos["x"] > goal_pos["x"] and "left" in safe_moves:
#         next_move = "left"
#     elif start_pos["y"] < goal_pos["y"] and "up" in safe_moves:
#         next_move = "up"
#     elif start_pos["y"] > goal_pos["y"] and "down" in safe_moves:
#         next_move = "down"
#     else:
#         next_move = random.choice(safe_moves)
#     return next_move
def get_naive_move(start_pos: Dict, goal_pos: Dict, safe_moves: List[str], game_state: Dict) -> Union[str, None]:
    """
    Get a move towards the goal while ensuring sufficient accessible area.
    Returns the best move that maintains maximum accessible area.
    """
    if not safe_moves:
        return None

    min_area_threshold = len(game_state['you']['body']) * 2
    max_accessible_area = -1
    best_move = None

    # Priority moves are those that move us closer to the goal
    priority_moves = []
    if start_pos["x"] < goal_pos["x"] and "right" in safe_moves:
        priority_moves.append("right")
    elif start_pos["x"] > goal_pos["x"] and "left" in safe_moves:
        priority_moves.append("left")

    if start_pos["y"] < goal_pos["y"] and "up" in safe_moves:
        priority_moves.append("up")
    elif start_pos["y"] > goal_pos["y"] and "down" in safe_moves:
        priority_moves.append("down")

    # First check priority moves
    for move in priority_moves:
        new_head = simulate_move(start_pos, move)
        area = flood_fill(new_head, game_state)
        print(f"Priority move {move} leads to area of {area}")
        if area >= min_area_threshold and area > max_accessible_area:
            max_accessible_area = area
            best_move = move

    # If no priority move gives sufficient area, check all safe moves
    if best_move is None:
        print("No priority moves meet area threshold, checking all safe moves")
        for move in safe_moves:
            new_head = simulate_move(start_pos, move)
            area = flood_fill(new_head, game_state)
            print(f"Safe move {move} leads to area of {area}")
            if area > max_accessible_area:
                max_accessible_area = area
                best_move = move

    return best_move



def simulate_move(head, move):
    if move == 'up':
        return {'x': head['x'], 'y': head['y'] + 1}
    elif move == 'down':
        return {'x': head['x'], 'y': head['y'] - 1}
    elif move == 'left':
        return {'x': head['x'] - 1, 'y': head['y']}
    elif move == 'right':
        return {'x': head['x'] + 1, 'y': head['y']}

def get_neighbors(head, board_width, board_height):
    neighbors = []
    directions = ['up', 'down', 'left', 'right']
    for move in directions:
        neighbor = simulate_move(head, move)
        if 0 <= neighbor['x'] < board_width and 0 <= neighbor['y'] < board_height:
            neighbors.append(neighbor)
    return neighbors


def flood_fill(start_position, game_state):
    board_width = game_state['board']['width']
    board_height = game_state['board']['height']
    occupied_area = set()
    for snake in game_state['board']['snakes']:
        for seg in snake['body']:
            occupied_area.add((seg['x'], seg['y']))

    visited = set()
    queue = deque()
    queue.append(start_position)
    visited.add((start_position['x'], start_position['y']))

    while queue:
        current = queue.popleft()
        neighbors = get_neighbors(current, board_width, board_height)

        for neighbor in neighbors:
            pos = (neighbor['x'], neighbor['y'])
            if pos not in visited and pos not in occupied_area:
                visited.add(pos)
                queue.append(neighbor)

    accessible_area = len(visited)
    return accessible_area


def choose_move_with_max_accessible_area(safe_moves, my_head, game_state):
    max_area = -1
    best_move = None
    for move in safe_moves:
        new_head = simulate_move(my_head, move)
        area = flood_fill(new_head, game_state)
        if area > max_area:
            max_area = area
            best_move = move
    return best_move if best_move else random.choice(safe_moves)




# Start server when `python main.py` is run
if __name__ == "__main__":
    from server import run_server

    run_server({"info": info, "start": start, "move": move, "end": end})


