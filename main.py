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
import typing


# info is called when you create your Battlesnake on play.battlesnake.com
# and controls your Battlesnake's appearance
# TIP: If you open your Battlesnake URL in a browser you should see this data
def info() -> typing.Dict:
    print("INFO")

    return {
        "apiversion": "1",
        "author": "cs5446-g37",
        "color": "#FAC3A4",
        "head": "smart-caterpillar",
        "tail": "mouse",
    }


# start is called when your Battlesnake begins a game
def start(game_state: typing.Dict):
    print("GAME START")


# end is called when your Battlesnake finishes a game
def end(game_state: typing.Dict):
    print("GAME OVER\n")


# move is called on every turn and returns your next move
# Valid moves are "up", "down", "left", or "right"
# See https://docs.battlesnake.com/api/example-move for available data
def move(game_state: typing.Dict) -> typing.Dict:

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

    if len(safe_moves) == 0:
        print(f"MOVE {game_state['turn']}: No safe moves detected! Moving down")
        return {"move": "down"}

    # Step 4 - Move towards food instead of random, to regain health and survive longer
    food = game_state['board']['food']

    # If we have safe moves and there's food, try to move towards closest food
    if len(safe_moves) > 0 and len(food) > 0:
        # Find closest food based on Manhattan distance between head and food
        closest_food = food[0]
        closest_distance = abs(my_head["x"] - food[0]["x"]) + abs(my_head["y"] - food[0]["y"])

        for f in food:
            distance = abs(my_head["x"] - f["x"]) + abs(my_head["y"] - f["y"])
            if distance < closest_distance:
                closest_food = f
                closest_distance = distance

        # Try to move towards the food if it's safe
        if my_head["x"] < closest_food["x"] and "right" in safe_moves:
            next_move = "right"
        elif my_head["x"] > closest_food["x"] and "left" in safe_moves:
            next_move = "left"
        elif my_head["y"] < closest_food["y"] and "up" in safe_moves:
            next_move = "up"
        elif my_head["y"] > closest_food["y"] and "down" in safe_moves:
            next_move = "down"
        else:
            next_move = random.choice(safe_moves)
    else:
        next_move = random.choice(safe_moves)

    print(f"MOVE {game_state['turn']}: {next_move}")
    return {"move": next_move}


# Start server when `python main.py` is run
if __name__ == "__main__":
    from server import run_server

    run_server({"info": info, "start": start, "move": move, "end": end})
