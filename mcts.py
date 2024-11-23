import math
import random
import time
from copy import deepcopy
from typing import Dict, List, Optional, Set, Tuple


class Node:
    def __init__(self, game_state: Dict, parent=None, action=None):
        self.state = game_state
        self.parent = parent
        self.action = action  # Action that led to this state
        self.children: Dict[str, 'Node'] = {}  # {action: Node}
        self.visits = 0
        self.value = 0.0
        self.unexpanded_actions = self._get_valid_actions()

    def _get_valid_actions(self) -> List[str]:
        """Get list of valid moves considering board boundaries and collisions."""
        valid_moves = ["up", "down", "left", "right"]
        my_head = self.state["you"]["head"]
        board_width = self.state["board"]["width"]
        board_height = self.state["board"]["height"]

        # Check board boundaries
        if my_head["x"] == 0:
            valid_moves.remove("left")
        if my_head["x"] == board_width - 1:
            valid_moves.remove("right")
        if my_head["y"] == 0:
            valid_moves.remove("down")
        if my_head["y"] == board_height - 1:
            valid_moves.remove("up")

        # Check self collisions
        next_positions = self._get_next_positions(my_head)
        for move in valid_moves.copy():
            if self._is_collision(next_positions[move], self.state):
                valid_moves.remove(move)

        return valid_moves

    def _get_valid_actions_for_snake(self, snake: Dict) -> List[str]:
        """Get list of valid moves for any snake."""
        valid_moves = ["up", "down", "left", "right"]
        head = snake["head"]
        board_width = self.state["board"]["width"]
        board_height = self.state["board"]["height"]

        # Check board boundaries
        if head["x"] == 0:
            valid_moves.remove("left")
        if head["x"] == board_width - 1:
            valid_moves.remove("right")
        if head["y"] == 0:
            valid_moves.remove("down")
        if head["y"] == board_height - 1:
            valid_moves.remove("up")

        # Check collisions
        next_positions = self._get_next_positions(head)
        for move in valid_moves.copy():
            # Create temporary state to check collision
            temp_state = deepcopy(self.state)
            # Replace the snake we're checking with a modified version
            for i, s in enumerate(temp_state["board"]["snakes"]):
                if s["id"] == snake["id"]:
                    temp_state["board"]["snakes"][i] = snake
                    break
            if self._is_collision(next_positions[move], temp_state):
                valid_moves.remove(move)

        return valid_moves

    def get_dangerous_positions(self) -> Set[Tuple[int, int]]:
        """Get positions that could be reached by longer enemy snakes in one move."""
        dangerous_positions = set()
        my_length = len(self.state["you"]["body"])

        for snake in self.state["board"]["snakes"]:
            # Skip our own snake
            if snake["id"] == self.state["you"]["id"]:
                continue

            # Only consider longer or equal snakes
            if len(snake["body"]) < my_length:
                continue

            # Get all possible next positions for this snake
            head = snake["head"]
            possible_moves = [
                (head["x"] + dx, head["y"] + dy)
                for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]
                if 0 <= head["x"] + dx < self.state["board"]["width"]
                   and 0 <= head["y"] + dy < self.state["board"]["height"]
            ]

            # Add all possible next positions to dangerous positions
            dangerous_positions.update(possible_moves)

        return dangerous_positions

    def _get_next_positions(self, pos: Dict[str, int]) -> Dict[str, Dict[str, int]]:
        """Get all possible next positions from current position."""
        return {
            "up": {"x": pos["x"], "y": pos["y"] + 1},
            "down": {"x": pos["x"], "y": pos["y"] - 1},
            "left": {"x": pos["x"] - 1, "y": pos["y"]},
            "right": {"x": pos["x"] + 1, "y": pos["y"]}
        }

    def _is_collision(self, pos: Dict[str, int], state: Dict, new_heads: Dict[str, Dict[str, int]] = None) -> bool:
        """
        Check if position collides with any snake body or new heads.
        new_heads: Dict mapping snake_id to new head position for simultaneous movement
        """
        # Check collisions with new heads from simultaneous movement
        if new_heads:
            # Head-to-head collisions
            head_positions = list(new_heads.values())
            if pos in head_positions and head_positions.count(pos) > 1:
                return True

        # Check collisions with snake bodies (excluding tails that will move)
        for snake in state["board"]["snakes"]:
            # For body collisions, exclude tail only if snake hasn't eaten
            # Would need to know if snake is eating food this turn for perfect accuracy
            for body_part in snake["body"][:-1]:
                if pos["x"] == body_part["x"] and pos["y"] == body_part["y"]:
                    return True
        return False

    def is_terminal(self) -> bool:
        """Check if current state is terminal (game over)."""
        # Game is over if snake is dead or no valid moves
        return (len(self.state["board"]["snakes"]) <= 1  # Use 0 when testing with only one snake!
                or
                self.state["you"]["health"] <= 0
                or
                not self._get_valid_actions()
                )

    def get_reward(self) -> float:
        """Calculate reward with strong emphasis on food-seeking behavior."""
        if self.is_terminal():
            return -1000.0  # Death is still heavily penalized

        reward = 0.0

        # Basic survival reward
        # reward += 50.0

        # Get distance to closest food
        closest_food_dist = self.get_closest_food_distance()
        if closest_food_dist is not None:
            # Strong reward for being close to food - this is now the primary component
            # We should NOT have this! This actually prevent the snake from moving towards food but wandering around food
            # reward = 1000.0 / (closest_food_dist + 1)

            # Extra reward for being very close to food
            # if closest_food_dist <= 2:
            #     reward += 50

            # Check if we moved closer to or further from food
            if self.parent is not None:
                prev_dist = self.parent.get_closest_food_distance()
                if prev_dist is not None:
                    if closest_food_dist < prev_dist:  # Moving closer to food
                        reward += 100.0
                    else:  # Moving away from food
                        reward -= 100.0
            pass

        # Huge reward for eating food
        if self.parent is not None:
            if self.state["you"]["health"] == 100 and self.parent.state["you"]["health"] < 100:
                reward += 200.0

            # Penalize moving towards dangerous positions
            dangerous_positions = self.parent.get_dangerous_positions()
            # print("danger positions")
            # print(dangerous_positions)
            head_pos = (self.state["you"]["head"]["x"], self.state["you"]["head"]["y"])
            # print("my head at", head_pos)
            if head_pos in dangerous_positions:
                # Significant penalty but not as severe as death
                # print("self state")
                # print(self.state)
                # print("parent state")
                # print(self.parent.state)
                # print("danger")
                reward -= 700.0

        return reward

    def get_closest_food_distance(self) -> Optional[float]:
        """Calculate Manhattan distance to closest food."""
        if not self.state["board"]["food"]:
            return None

        head = self.state["you"]["head"]
        return min(
            abs(food["x"] - head["x"]) + abs(food["y"] - head["y"])
            for food in self.state["board"]["food"]
        )


class MCTS:
    def __init__(self, time_limit: float = 0.1, rollout_limit: int = 50):
        self.time_limit = time_limit
        self.rollout_limit = rollout_limit
        self.exploration_constant = 100  # This large exploration constant is due to large reward values

    def search(self, initial_state: Dict) -> str:
        root = Node(initial_state)
        end_time = time.time() + self.time_limit

        iter_count = 0
        while time.time() < end_time:
            # print(f"iter: {iter_count}")
            iter_count += 1
            node = self._select(root)
            if not node.is_terminal():
                node = self._expand(node)
                reward = self._rollout(node)
                self._backpropagate(node, reward)
            else:
                self._backpropagate(node, node.get_reward())
            # print("\n")

        # Select best action based on average value
        best_action = self._get_best_action(root)
        print(f"Select best action {best_action} after {iter_count} iterations\n")
        return best_action

    def _select(self, node: Node) -> Node:
        """Select node to expand using UCT."""
        while not node.is_terminal():
            if node.unexpanded_actions:
                return node

            node = self._get_best_uct_child(node)
        return node

    def _get_best_uct_child(self, node: Node) -> Node:
        """Get child with highest UCT value."""
        best_score = float('-inf')
        best_child = None

        for child in node.children.values():
            # UCT formula: vi + C * sqrt(ln(N) / ni)
            exploitation = child.value / child.visits if child.visits > 0 else 0
            exploration = math.sqrt(math.log(node.visits) / child.visits) if child.visits > 0 else float('inf')
            uct_score = exploitation + self.exploration_constant * exploration

            if uct_score > best_score:
                best_score = uct_score
                best_child = child

        return best_child

    # def _apply_simultaneous_moves(self, state: Dict, moves: Dict[str, str]) -> None:
    #     """Apply all snake moves simultaneously."""
    #     # First collect all new head positions
    #     new_heads = {}
    #     snakes_eating = set()  # Track which snakes are eating food
    #
    #     for snake_id, action in moves.items():
    #         # Find the snake
    #         snake = None
    #         for s in state["board"]["snakes"]:
    #             if s["id"] == snake_id:
    #                 snake = s
    #                 break
    #         if not snake:
    #             continue
    #
    #         # Calculate new head position
    #         new_head = self._get_next_position(snake["head"], action)
    #         new_heads[snake_id] = new_head
    #
    #         # Check if snake will eat food
    #         if new_head in state["board"]["food"]:
    #             snakes_eating.add(snake_id)
    #
    #     # Check for head-to-head collisions and remove losing snakes
    #     # In head-to-head, longer snake wins
    #     head_positions = {}  # position -> list of (snake_id, length)
    #     for snake_id, new_head in new_heads.items():
    #         pos_str = f"{new_head['x']},{new_head['y']}"
    #         snake = next(s for s in state["board"]["snakes"] if s["id"] == snake_id)
    #         head_positions.setdefault(pos_str, []).append((snake_id, len(snake["body"])))
    #
    #     # Remove snakes that lose head-to-head collisions
    #     for pos_list in head_positions.values():
    #         if len(pos_list) > 1:
    #             # Find max length
    #             max_length = max(length for _, length in pos_list)
    #             # Remove all snakes of non-max length
    #             for snake_id, length in pos_list:
    #                 if length <= max_length:
    #                     new_heads.pop(snake_id)
    #
    #     # Now update all surviving snakes simultaneously
    #     for i, snake in enumerate(state["board"]["snakes"]):
    #         if snake["id"] not in new_heads:
    #             # Snake died in head-to-head collision
    #             state["board"]["snakes"].pop(i)
    #             continue
    #
    #         new_head = new_heads[snake["id"]]
    #
    #         # Update snake
    #         snake["body"].insert(0, new_head)
    #         snake["head"] = new_head
    #
    #         # Handle food
    #         if snake["id"] in snakes_eating:
    #             state["board"]["food"].remove(new_head)
    #             snake["health"] = 100
    #         else:
    #             snake["body"].pop()
    #             snake["health"] -= 1

    def _get_valid_moves_for_snake(self, snake: Dict, state: Dict) -> List[str]:
        """Get valid moves for a given snake."""
        valid_moves = ["up", "down", "left", "right"]
        head = snake["head"]

        # Check board boundaries
        if head["x"] == 0:
            valid_moves.remove("left")
        if head["x"] == state["board"]["width"] - 1:
            valid_moves.remove("right")
        if head["y"] == 0:
            valid_moves.remove("down")
        if head["y"] == state["board"]["height"] - 1:
            valid_moves.remove("up")

        # Check collisions with snake bodies
        for move in valid_moves.copy():
            next_pos = {
                "x": head["x"] + (1 if move == "right" else -1 if move == "left" else 0),
                "y": head["y"] + (1 if move == "up" else -1 if move == "down" else 0)
            }

            for other_snake in state["board"]["snakes"]:
                for body_part in other_snake["body"][:-1]:  # Exclude tail
                    if next_pos["x"] == body_part["x"] and next_pos["y"] == body_part["y"]:
                        if move in valid_moves:
                            valid_moves.remove(move)
                        break

        return valid_moves

    def _choose_opponent_move(self, snake: Dict, valid_moves: List[str], state: Dict) -> str:
        """Choose move for opponent snake using simple heuristics."""
        if not valid_moves:
            return "up"  # Default move if no valid moves

        head = snake["head"]

        # If health is low, try to move towards closest food
        if snake["health"] < 50 and state["board"]["food"]:
            closest_food = min(
                state["board"]["food"],
                key=lambda food: abs(food["x"] - head["x"]) + abs(food["y"] - head["y"])
            )

            # Try to move towards food
            if closest_food["x"] > head["x"] and "right" in valid_moves:
                return "right"
            if closest_food["x"] < head["x"] and "left" in valid_moves:
                return "left"
            if closest_food["y"] > head["y"] and "up" in valid_moves:
                return "up"
            if closest_food["y"] < head["y"] and "down" in valid_moves:
                return "down"

        return random.choice(valid_moves)

        # # If we're bigger than nearby snakes, try to move towards their heads
        # my_length = len(snake["body"])
        # for other_snake in state["board"]["snakes"]:
        #     if other_snake["id"] != snake["id"] and len(other_snake["body"]) < my_length:
        #         other_head = other_snake["head"]
        #         dist = abs(head["x"] - other_head["x"]) + abs(head["y"] - other_head["y"])
        #
        #         if dist <= 2:  # Only chase if close enough
        #             if other_head["x"] > head["x"] and "right" in valid_moves:
        #                 return "right"
        #             if other_head["x"] < head["x"] and "left" in valid_moves:
        #                 return "left"
        #             if other_head["y"] > head["y"] and "up" in valid_moves:
        #                 return "up"
        #             if other_head["y"] < head["y"] and "down" in valid_moves:
        #                 return "down"
        #
        # # Otherwise, choose random move with slight preference for center
        # center_x = state["board"]["width"] / 2
        # center_y = state["board"]["height"] / 2
        #
        # moves_with_scores = []
        # for move in valid_moves:
        #     next_pos = {
        #         "x": head["x"] + (1 if move == "right" else -1 if move == "left" else 0),
        #         "y": head["y"] + (1 if move == "up" else -1 if move == "down" else 0)
        #     }
        #
        #     # Score based on distance to center (higher score = closer to center)
        #     center_dist = abs(next_pos["x"] - center_x) + abs(next_pos["y"] - center_y)
        #     score = 1.0 / (center_dist + 1)
        #     moves_with_scores.append((move, score))
        #
        # # Choose randomly with weights based on scores
        # total_score = sum(score for _, score in moves_with_scores)
        # if total_score == 0:
        #     return random.choice(valid_moves)
        #
        # r = random.random() * total_score
        # cumulative = 0
        # for move, score in moves_with_scores:
        #     cumulative += score
        #     if cumulative >= r:
        #         return move
        #
        # return valid_moves[-1]

    def _apply_action(self, state: Dict, action: str) -> None:
        """Apply action to state with death checks and opponent movement."""
        # Move our snake
        head = state["you"]["head"]
        new_head = {
            "x": head["x"] + (1 if action == "right" else -1 if action == "left" else 0),
            "y": head["y"] + (1 if action == "up" else -1 if action == "down" else 0)
        }

        # Update our snake body
        state["you"]["body"].insert(0, new_head)
        state["you"]["head"] = new_head

        # Handle food for our snake
        if new_head in state["board"]["food"]:
            state["board"]["food"].remove(new_head)
            state["you"]["health"] = 100
        else:
            state["you"]["body"].pop()
            state["you"]["health"] -= 1

        # Move other snakes with simple heuristics
        for snake in state["board"]["snakes"]:
            if snake["id"] != state["you"]["id"]:
                # Get valid moves for opponent
                valid_moves = self._get_valid_moves_for_snake(snake, state)
                if not valid_moves:
                    continue  # Snake has no valid moves, will die

                # Choose move based on simple heuristics
                move = self._choose_opponent_move(snake, valid_moves, state)

                # Apply opponent move
                opponent_head = snake["head"]
                new_opponent_head = {
                    "x": opponent_head["x"] + (1 if move == "right" else -1 if move == "left" else 0),
                    "y": opponent_head["y"] + (1 if move == "up" else -1 if move == "down" else 0)
                }

                snake["body"].insert(0, new_opponent_head)
                snake["head"] = new_opponent_head

                # Handle food for opponent
                if new_opponent_head in state["board"]["food"]:
                    state["board"]["food"].remove(new_opponent_head)
                    snake["health"] = 100
                else:
                    snake["body"].pop()
                    snake["health"] -= 1

        # Check for snake deaths
        surviving_snakes = []
        for snake in state["board"]["snakes"]:
            is_dead = False
            head = snake["head"]

            # Check wall collision
            if (head["x"] < 0 or head["x"] >= state["board"]["width"] or
                    head["y"] < 0 or head["y"] >= state["board"]["height"]):
                is_dead = True

            # Check body collisions with all snakes
            for other_snake in state["board"]["snakes"]:
                # Check collision with other snake's body (including head)
                for i, body_part in enumerate(other_snake["body"]):
                    if head["x"] == body_part["x"] and head["y"] == body_part["y"]:
                        # Head-to-head collision
                        if (i == 0 and snake["id"] != other_snake["id"] and
                                len(snake["body"]) <= len(other_snake["body"])):
                            is_dead = True
                            break
                        # Body collision
                        elif i > 0:
                            is_dead = True
                            break
                if is_dead:
                    break

            # Check health
            if snake["health"] <= 0:
                is_dead = True

            if not is_dead:
                surviving_snakes.append(snake)
            else:
                # Our snake has dead
                if snake["id"] == state["you"]["id"]:
                    state["you"]["health"] = 0

        # Update list of surviving snakes
        state["board"]["snakes"] = surviving_snakes

        # Update our snake in the snakes list
        for i, snake in enumerate(state["board"]["snakes"]):
            if snake["id"] == state["you"]["id"]:
                state["board"]["snakes"][i] = state["you"]
                break

    def _expand(self, node: Node) -> Node:
        """Create a new child node."""
        my_action = random.choice(node.unexpanded_actions)
        node.unexpanded_actions.remove(my_action)

        # Create new state by applying actions
        new_state = deepcopy(node.state)

        # Collect moves for all snakes
        # moves = {node.state["you"]["id"]: my_action}

        # Get random moves for other snakes
        # for snake in new_state["board"]["snakes"]:
        #     if snake["id"] != new_state["you"]["id"]:
        #         temp_node = Node(new_state)
        #         valid_moves = temp_node._get_valid_actions_for_snake(snake)
        #         if valid_moves:
        #             moves[snake["id"]] = random.choice(valid_moves)
        #         else:
        #             moves[snake["id"]] = "up"

        # Apply all moves simultaneously
        self._apply_action(new_state, my_action)

        # Create and store new node
        child = Node(new_state, parent=node, action=my_action)
        node.children[my_action] = child
        return child

    def _rollout(self, node: Node) -> float:
        state = deepcopy(node.state)
        current_node = Node(state, parent=deepcopy(node.parent))
        depth = 0
        total_reward = 0
        discount_factor = 0.9  # Favor earlier rewards

        while not current_node.is_terminal() and depth < self.rollout_limit:
            # print(depth)
            total_reward += current_node.get_reward() * (discount_factor ** depth)

            # Collect moves for all snakes
            moves = {}

            # Get rollout policy move for our snake
            # moves[state["you"]["id"]] = self._rollout_policy(current_node)
            my_action = self._rollout_policy(current_node)

            # Get random moves for other snakes
            # for snake in state["board"]["snakes"]:
            #     if snake["id"] != state["you"]["id"]:
            #         temp_node = Node(state)
            #         valid_moves = temp_node._get_valid_actions_for_snake(snake)
            #         if valid_moves:
            #             moves[snake["id"]] = random.choice(valid_moves)
            #         else:
            #             moves[snake["id"]] = "up"

            # Apply all moves simultaneously
            new_state = deepcopy(state)
            self._apply_action(new_state, my_action)

            state = new_state
            current_node = Node(new_state, parent=current_node)
            depth += 1

        # Add final state reward
        total_reward += current_node.get_reward() * (discount_factor ** depth)
        return total_reward

    def _rollout_policy(self, node: Node) -> str:
        """Aggressive food-seeking rollout policy."""
        valid_actions = node._get_valid_actions()
        if not valid_actions:
            return "up"  # Default move if no valid moves

        # If there's food, always try to move towards it
        if node.state["board"]["food"]:
            head = node.state["you"]["head"]
            closest_food = min(
                node.state["board"]["food"],
                key=lambda food: abs(food["x"] - head["x"]) + abs(food["y"] - head["y"])
            )

            # Calculate distances for each move
            distances = {}
            for action in valid_actions:
                next_pos = self._get_next_position(head, action)
                distances[action] = abs(next_pos["x"] - closest_food["x"]) + abs(next_pos["y"] - closest_food["y"])

            # Return the action that gets us closest to food
            return min(distances.items(), key=lambda x: x[1])[0]

        return random.choice(valid_actions)

    def _get_next_position(self, pos: Dict[str, int], action: str) -> Dict[str, int]:
        """Get next position given current position and action."""
        if action == "up":
            return {"x": pos["x"], "y": pos["y"] + 1}
        elif action == "down":
            return {"x": pos["x"], "y": pos["y"] - 1}
        elif action == "left":
            return {"x": pos["x"] - 1, "y": pos["y"]}
        else:  # right
            return {"x": pos["x"] + 1, "y": pos["y"]}

    def _get_best_action(self, root: Node) -> str:
        """Select best action based on average value and distance to food."""
        if not root.children:
            return "up"  # Default move if no children

        # If there's food, factor in distance to food when selecting best action
        if root.state["board"]["food"]:
            head = root.state["you"]["head"]
            closest_food = min(
                root.state["board"]["food"],
                key=lambda food: abs(food["x"] - head["x"]) + abs(food["y"] - head["y"])
            )

            best_score = float('-inf')
            best_action = "up"

            for action, child in root.children.items():
                # Calculate next position for this action
                next_pos = self._get_next_position(head, action)
                distance_to_food = abs(next_pos["x"] - closest_food["x"]) + abs(next_pos["y"] - closest_food["y"])

                # Combine MCTS value with distance to food
                mcts_score = child.value / child.visits if child.visits > 0 else float('-inf')
                distance_score = 1500.0 / (distance_to_food + 1)  # Normalize distance score
                combined_score = (0.5 * mcts_score) + (0.5 * distance_score)

                print(
                    f"Action [{child.action}]: visits={child.visits} mcts_score={mcts_score}\t distance_score={distance_score}\t combined_score={combined_score}")

                if combined_score > best_score:
                    best_score = combined_score
                    best_action = action

            return best_action

        # If no food, fall back to standard MCTS selection
        return max(root.children.items(), key=lambda x: x[1].value / x[1].visits if x[1].visits > 0 else float('-inf'))[0]

    # def _get_food_seeking_action(self, node: Node, valid_actions: List[str]) -> Optional[str]:
    #     """Choose action that moves closer to nearest food."""
    #     if not node.state["board"]["food"]:
    #         return None
    #
    #     head = node.state["you"]["head"]
    #     closest_food = min(
    #         node.state["board"]["food"],
    #         key=lambda food: abs(food["x"] - head["x"]) + abs(food["y"] - head["y"])
    #     )
    #
    #     # Try to move towards food
    #     if closest_food["x"] > head["x"] and "right" in valid_actions:
    #         return "right"
    #     if closest_food["x"] < head["x"] and "left" in valid_actions:
    #         return "left"
    #     if closest_food["y"] > head["y"] and "up" in valid_actions:
    #         return "up"
    #     if closest_food["y"] < head["y"] and "down" in valid_actions:
    #         return "down"
    #
    #     return None

    def _backpropagate(self, node: Node, reward: float) -> None:
        """Update values up the tree."""
        while node:
            node.visits += 1
            node.value += reward
            node = node.parent

    # def _get_best_action(self, root: Node) -> str:
    #     """Select best action based on average value."""
    #     best_value = float('-inf')
    #     best_action = "up"  # Default move
    #     print(root.children.items())
    #     for action, child in root.children.items():
    #         print(action, child.value, child.visits)
    #         child_value = child.value / child.visits if child.visits > 0 else float('-inf')
    #         if child_value > best_value:
    #             best_value = child_value
    #             best_action = action
    #
    #     return best_action


def move(game_state: Dict) -> Dict:
    """Main move function called by the game."""
    mcts = MCTS(time_limit=0.3)  # 300ms time limit
    next_move = mcts.search(game_state)
    return {"move": next_move}
