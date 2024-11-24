from random import choice
class MonteCarloSearchTree:
    def __init__(self, max_iterations):
        self.max_iterations = max_iterations
    
        
    def find_best_action(self, initial_state):
        root = GameNode(initial_state, None)
        for i  in range(self.max_iterations):
            selected_node = self.traverse(root)
            terminal_node = selected_node
    
            while not terminal_node.is_terminal():
                possible_actions = terminal_node.get_valid_actions()
                terminal_node= terminal_node.apply_action(choice(possible_actions))
            reward = terminal_node.calculate_reward()

            while selected_node is not None:
                selected_node.visit_count+= 1
                selected_node.total_reward+=reward
                selected_node =selected_node.parent
        best_action = max(root.children, key=lambda action: root.children[action].visit_count)
        return best_action

    def traverse(self, node):
        while not node.is_terminal():
            if node.is_fully_expanded:
                node = self.select_child(node)
            else:
                return self.expand(node)
        return node


    def select_child(self, node,exploration_constant=1.414):
        import math
        best_value = float('-inf')
        best_children = []
        for child in node.children.values():
            ucb_value = (child.total_reward / child.visit_count +
                         exploration_constant * math.sqrt(math.log(node.visit_count) / child.visit_count))
            if ucb_value > best_value:
                best_value = ucb_value
                best_children = [child]
            elif ucb_value == best_value:
                best_children.append(child)

        from random import choice
        return choice(best_children)

    def expand(self, node):
        untried_actions = [action for action in node.get_valid_actions() if action not in node.children]
        action = untried_actions.pop()
        child_node = node.apply_action(action)
        node.children[action] = child_node
        if not untried_actions:
            node.is_fully_expanded = True
        return child_node
    
class GameNode:
    def __init__(self, state, parent_node):
        self.game_state = state
        self.parent = parent_node
        self.children = dict()
        self.visit_count =0
        self.total_reward = 0
        self.is_fully_expanded = False

    def calculate_reward(self):
        #Calculate the reward based on whether the game is in a terminal state.
        if self.is_terminal():
            return self.game_state['turn']+self.game_state['you']['length']
        return 1e4/self.get_min_dis()  # Arbitrary high reward for non-terminal states
    
    def get_min_dis(self):
        head = self.game_state['you']['head']
        food_list = self.game_state['board']['food']
        if not food_list:
            return float('inf')    
        min_distance = float('inf')
        for food in food_list:
            distance = abs(food['x'] - head['x']) + abs(food['y'] - head['y'])
            if distance < min_distance:
                min_distance = distance
        return min_distance

    def is_terminal(self):
        # Check if the game has reached a terminal state.
        return len(self.get_valid_actions()) == 0 or self.game_state['you']['health'] == 0

    def apply_action(self, action):
        #Generate the resulting game state after applying a specific action.
        from copy import deepcopy
        new_state = deepcopy(self.game_state)
        head = new_state['you']['head']

        # Define movement directions
        movement = {'up': (0, 1), 'down': (0, -1), 'left': (-1, 0), 'right': (1, 0)}
        if action not in movement:
            raise ValueError(f"Invalid action: {action}")

        head['x'] += movement[action][0]
        head['y'] += movement[action][1]
        new_state['turn'] += 1

        # Handle food consumption
        if head in new_state['board']['food']:
            new_state['board']['food'].remove(head)
            new_state['you']['health'] = 100
            new_state['you']['length']+=1
        else:
            new_state['you']['health'] -= 1

        # Update snake's body
        new_state['you']['body'].insert(0, deepcopy(head))
        new_state['you']['body'].pop()

        # Update the board to reflect the new state of the snake
        for snake in new_state['board']['snakes']:
            if snake['id'] == new_state['you']['id']:
                snake.update(new_state['you'])
                break

        return GameNode(new_state, self)

    def get_valid_actions(self):
        """Determine all valid actions that the snake can take."""
        possible_actions = ['up', 'down', 'left', 'right']
        head = self.game_state['you']['head']
        width, height = self.game_state['board']['width'], self.game_state['board']['height']

        if head['x'] == 0:
            possible_actions.remove('left')
        if head['x'] == width - 1:
            possible_actions.remove('right')
        if head['y'] == 0:
            possible_actions.remove('down')
        if head['y'] == height - 1:
            possible_actions.remove('up')

        # Check for collisions with other snakes' bodies
        all_snake_positions = [pos for snake in self.game_state['board']['snakes'] for pos in snake['body']]
        movement = {'up': (0, 1), 'down': (0, -1), 'left': (-1, 0), 'right': (1, 0)}
        for action in possible_actions[:]:
            next_position = {'x': head['x'] + movement[action][0], 'y': head['y'] + movement[action][1]}
            if next_position in all_snake_positions:
                possible_actions.remove(action)

        return possible_actions
