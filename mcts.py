import math
import random
import time
from copy import deepcopy
from typing import Dict, List, Optional


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

    def _get_next_positions(self, pos: Dict[str, int]) -> Dict[str, Dict[str, int]]:
        """Get all possible next positions from current position."""
        return {
            "up": {"x": pos["x"], "y": pos["y"] + 1},
            "down": {"x": pos["x"], "y": pos["y"] - 1},
            "left": {"x": pos["x"] - 1, "y": pos["y"]},
            "right": {"x": pos["x"] + 1, "y": pos["y"]}
        }

    def _is_collision(self, pos: Dict[str, int], state: Dict) -> bool:
        """Check if position collides with any snake body."""
        for snake in state["board"]["snakes"]:
            for body_part in snake["body"][:-1]:  # Exclude tail as it will move
                if pos["x"] == body_part["x"] and pos["y"] == body_part["y"]:
                    return True
        return False

    def is_terminal(self) -> bool:
        """Check if current state is terminal (game over)."""
        # Game is over if snake is dead or no valid moves
        return (len(self.state["board"]["snakes"]) <= 1  # Use 0 when testing with only one snake!
                or
                self.state["you"]["health"] <= 0 or
                not self._get_valid_actions())

    def get_reward(self) -> float:
        """Calculate reward for current state."""
        if self.state["you"]["health"] <= 0:
            return -1000.0  # Heavy penalty for death

        reward = 0.0

        # Penalize for reduced health to encourage the snake to find food
        # reward = float(self.state["you"]["health"]) / 10.0
        reward -= (100 - self.state["you"]["health"])

        # Reward for length
        reward += len(self.state["you"]["body"]) * 2

        # Reward for consuming some food and resuming full health
        if self.state["you"]["health"] == 100:
            reward += 50.0

        # Reward for being close to food when health is low
        if self.state["you"]["health"] < 50:
            closest_food_dist = self.get_closest_food_distance()
            if closest_food_dist is not None:
                reward += (1000.0 / (closest_food_dist + 1))

        # Small penalization if did not move towards the food next to it
        if self.parent is not None:
            if self.parent.get_closest_food_distance() == 1 and self.state["you"]["health"] != 100:
                reward -= 50.0

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
            iter_count += 1
            node = self._select(root)
            if not node.is_terminal():
                node = self._expand(node)
                reward = self._rollout(node)
                self._backpropagate(node, reward)
            else:
                self._backpropagate(node, node.get_reward())

        # Select best action based on average value
        best_action = self._get_best_action(root)
        print(f"Select best action {best_action} after {iter_count} iterations")
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

    def _expand(self, node: Node) -> Node:
        """Create a new child node."""
        action = random.choice(node.unexpanded_actions)
        node.unexpanded_actions.remove(action)

        # Create new state by applying action
        new_state = deepcopy(node.state)
        self._apply_action(new_state, action)

        # Create and store new node
        child = Node(new_state, parent=node, action=action)
        node.children[action] = child
        return child

    def _apply_action(self, state: Dict, action: str) -> None:
        """Apply action to state."""
        head = state["you"]["head"]
        new_head = {
            "x": head["x"] + (1 if action == "right" else -1 if action == "left" else 0),
            "y": head["y"] + (1 if action == "up" else -1 if action == "down" else 0)
        }

        # Update snake body
        state["you"]["body"].insert(0, new_head)
        state["you"]["head"] = new_head

        # Handle food
        if new_head in state["board"]["food"]:
            state["board"]["food"].remove(new_head)
            state["you"]["health"] = 100
        else:
            state["you"]["body"].pop()
            state["you"]["health"] -= 1

        # Update snake in board snakes list
        for i, snake in enumerate(state["board"]["snakes"]):
            if snake["id"] == state["you"]["id"]:
                state["board"]["snakes"][i] = state["you"]
                break

    # def _rollout(self, node: Node) -> float:
    #     """Simulate game from node until terminal state."""
    #     state = deepcopy(node.state)
    #     current_node = Node(state)
    #     depth = 0
    #
    #     while not current_node.is_terminal() and depth < self.rollout_limit:
    #         action = self._rollout_policy(current_node)
    #         self._apply_action(state, action)
    #         current_node = Node(state)
    #         depth += 1
    #
    #     return current_node.get_reward()

    # Use sum of rewards during rollout instead of the reward from terminal node
    def _rollout(self, node: Node) -> float:
        state = deepcopy(node.state)
        current_node = Node(state, parent=node)
        depth = 0
        total_reward = 0
        discount_factor = 0.95  # Favor earlier rewards

        while not current_node.is_terminal() and depth < self.rollout_limit:
            total_reward += current_node.get_reward() * (discount_factor ** depth)
            action = self._rollout_policy(current_node)
            self._apply_action(state, action)
            current_node = Node(state, parent=current_node)
            depth += 1

        # Add final state reward
        total_reward += current_node.get_reward() * (discount_factor ** depth)
        return total_reward

    def _rollout_policy(self, node: Node) -> str:
        """Simple rollout policy using heuristics."""
        valid_actions = node._get_valid_actions()
        if not valid_actions:
            return "up"  # Default move if no valid moves

        # Prioritize food when health is low
        # if node.state["you"]["health"] < 50:
        best_action = self._get_food_seeking_action(node, valid_actions)
        if best_action:
            return best_action

        # Otherwise choose random valid action
        return random.choice(valid_actions)

    def _get_food_seeking_action(self, node: Node, valid_actions: List[str]) -> Optional[str]:
        """Choose action that moves closer to nearest food."""
        if not node.state["board"]["food"]:
            return None

        head = node.state["you"]["head"]
        closest_food = min(
            node.state["board"]["food"],
            key=lambda food: abs(food["x"] - head["x"]) + abs(food["y"] - head["y"])
        )

        # Try to move towards food
        if closest_food["x"] > head["x"] and "right" in valid_actions:
            return "right"
        if closest_food["x"] < head["x"] and "left" in valid_actions:
            return "left"
        if closest_food["y"] > head["y"] and "up" in valid_actions:
            return "up"
        if closest_food["y"] < head["y"] and "down" in valid_actions:
            return "down"

        return None

    def _backpropagate(self, node: Node, reward: float) -> None:
        """Update values up the tree."""
        while node:
            node.visits += 1
            node.value += reward
            node = node.parent

    def _get_best_action(self, root: Node) -> str:
        """Select best action based on average value."""
        best_value = float('-inf')
        best_action = "up"  # Default move
        print(root.children.items())
        for action, child in root.children.items():
            print(action, child.value, child.visits)
            child_value = child.value / child.visits if child.visits > 0 else float('-inf')
            if child_value > best_value:
                best_value = child_value
                best_action = action

        return best_action


def move(game_state: Dict) -> Dict:
    """Main move function called by the game."""
    mcts = MCTS(time_limit=0.1)  # 100ms time limit
    next_move = mcts.search(game_state)
    return {"move": next_move}
