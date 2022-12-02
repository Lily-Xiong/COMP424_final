# Student agent: Add your own agent here
from agents.agent import Agent
from store import register_agent
import numpy as np
import sys
import math
import random
from copy import deepcopy


@register_agent("student_agent")
class StudentAgent(Agent):
    """
    A dummy class for your implementation. Feel free to use this class to
    add any helper functionalities needed for your agent.
    """

    def __init__(self):
        super(StudentAgent, self).__init__()
        self.name = "StudentAgent"
        self.dir_map = {
            "u": 0,
            "r": 1,
            "d": 2,
            "l": 3,
        }
        self.moves = ((-1, 0), (0, 1), (1, 0), (0, -1))
        self.autoplay = True

    def step(self, chess_board, my_pos, adv_pos, max_step):
        """
        Implement the step function of your agent here.
        You can use the following variables to access the chess board:
        - chess_board: a numpy array of shape (x_max, y_max, 4)
        - my_pos: a tuple of (x, y)
        - adv_pos: a tuple of (x, y)
        - max_step: an integer

        You should return a tuple of ((x, y), dir),
        where (x, y) is the next position of your agent and dir is the direction of the wall
        you want to put on.

        Please check the sample implementation in agents/random_agent.py or agents/human_agent.py for more details.
        """
        #General Idea for the main algo:
        
        # 1. create a new node to represent the current state
        # 2. Make the new node the root of the tree
        # 3. select the best child in tree 
        # 4. when leaf node is reached, if game does not end, expand 
        # 5. from expanded node, simulate the game, for both our player and adversary. Return game result
        # 6. backpropagate - updates the visit count and score of the nodes visited during selection and expansion
        
        return my_pos, self.dir_map["u"]



#Class representing the tree for Monte Carlo Search
class MonteCarloSearchTree():
    def __init__(self, move, rootNode):
        self.rootNode = rootNode


#Class representing one node in the Monte Carlo Search Tree
class TreeNode:
    def __init__(self, chessboard, my_pos, adv_pos, parentNode=None):
        self.parent = parentNode
        self.children = []
        self.num_of_visit = 0
        self.num_of_wins = 0
        self.chessboard = chessboard
        self.my_pos = my_pos
        self.adv_pos = adv_pos

   #Select the best node during tree traversal
    def select_best_node(self):
        bestNode = self

        while not self.is_terminal():
            bestNode = self.find_best_child_node_by_uct(bestNode)

        return bestNode

    #Expand the node by adding one random node as its child
    def expandNode(self, chess_board, max_step, adv_pos):
        parent_node = self
        new_pos = getNextPossibleMove(chess_board, self.pos, adv_pos, )
        new_node = TreeNode(parent_node, )
    
    #Simulate one game from a given node
    def simulation(self, max_step, we_first_or_second):
        # returns the score after the game has ended
        # if we_first_or_second = 0 we play first
        # if we_first_or_second = 1 we play second
        # check end game

        # use copies, so we don't change information for that node
        my_pos_copy = self.my_pos
        adv_pos_copy = self.adv_pos
        chess_board_copy = self.chessboard

        turn = we_first_or_second
        board_size = len(self.chessboard[0])
        # results[0] p0 score, results[2] p0 score
        # TODO check if the position of parameter of mypocopy and advposcopy changes when we first or second changes
        results = check_endgame(chess_board_copy, len(self.chessboard[0]), my_pos_copy, adv_pos_copy)

        # while game has not ended
        while not results[0]:
            if turn == 0:
                # get new random move
                my_new_pos, my_new_dir = random_move(chess_board_copy, my_pos_copy, adv_pos_copy, max_step)
                # TODO check
                #set barrier on chessboard copy
                chess_board_copy = set_barrier(chess_board_copy, my_new_pos[0], my_new_pos[2], my_new_dir)
                # change my position
                my_pos_copy = my_new_pos
                # change turn to adv
                turn = 1

            elif turn == 1:
                adv_new_pos, adv_new_dir = random_move(chess_board_copy, adv_pos_copy, my_pos_copy, max_step)
                # TODO check
                chess_board_copy = set_barrier(chess_board_copy, adv_new_pos[0], adv_new_pos[2], adv_new_dir)
                adv_pos_copy = adv_new_pos
                turn = 0

            # check results
            results = check_endgame(chess_board_copy, len(self.chessboard[0]), my_pos_copy, adv_pos_copy)

        # if adv wins return -1
        if results[2] > results[1]:
            score = -1

        elif results[1] == results[1]:
            score = 0

        else:
            score = 1

        return score

    
    #Backpropagate on the nodes based on the result of simulation
    def backpropagation(self, gameResult):
        currentNode = self
        while (currentNode != None):
            currentNode.update_data(gameResult)
            currentNode = currentNode.parent


     # ---- HELPER FUNCTIONS -------

    # select the best child node using UCT
    def find_best_child_node_by_uct(self):
        #a function to calculate UCT Value
        max_child = self.get_max_uct_children()
        return max_child

    def get_max_uct_children(self):

        # array that stores UCT value for each child of the current node
        UCT_arr = []

        # iterate through the list of children and calculate the UCT for each, and
        # return the child that has the max UCT value
        for i in range(len(self.children)):
            UCT_arr.append(self.children[i].uct)

        max_UCT = max(UCT_arr)
        max_UCT_index = UCT_arr.index(max_UCT)

        return self.children[max_UCT_index]

    #Calculate the UCT value for a node
    def uct(self):
        if self.parent == None:
            parent_visits = 0
        else:
            parent_visits = self.parent.num_of_visit
        return (self.num_of_wins / self.num_of_visit) + math.sqrt(2) * (math.sqrt(
            math.log(parent_visits) / self.num_of_visit))

    #Update the node's number of visit and win/lose
    def update_data(self, game_result):
        self.num_of_visit += 1
        if game_result == "win":
            self.num_of_wins += 1

    #Check if a node is terminal
    def is_terminal(self):
        if len(self.children) == 0:
            return True
        return False

    #Get a possible move from the current node/state of the board 
    def get_next_possible_move(chess_board, my_pos, adv_pos, max_step):
        move = generate_random_move(my_pos, max_step)
        ((x,y),dir) = move
        new_pos = (x,y)
        while check_valid_step(chess_board, adv_pos, my_pos, new_pos, dir, max_step) == False:
            move = generate_random_move(my_pos, max_step)

        return move

    #Pick a random move that can be made from the current state/ node
    def generate_random_move(self, max_step):
        x, y = self.my_pos
        random_x = random.randint(x - max_step, x + max_step)
        random_y = random.randint(x - max_step, x + max_step)
        random_dir = random.randint(0, 3)

        return ((random_x, random_y), random_dir)








def random_move(chess_board, my_pos, adv_pos, max_step):
    # Moves (Up, Right, Down, Left)
    ori_pos = deepcopy(my_pos)
    moves = ((-1, 0), (0, 1), (1, 0), (0, -1))
    steps = np.random.randint(0, max_step + 1)

    # Random Walk
    for _ in range(steps):
        r, c = my_pos
        dir = np.random.randint(0, 4)
        m_r, m_c = moves[dir]
        my_pos = (r + m_r, c + m_c)

        # Special Case enclosed by Adversary
        k = 0
        while chess_board[r, c, dir] or my_pos == adv_pos:
            k += 1
            if k > 300:
                break
            dir = np.random.randint(0, 4)
            m_r, m_c = moves[dir]
            my_pos = (r + m_r, c + m_c)

        if k > 300:
            my_pos = ori_pos
            break

    # Put Barrier
    dir = np.random.randint(0, 4)
    r, c = my_pos
    while chess_board[r, c, dir]:
        dir = np.random.randint(0, 4)

    return my_pos, dir

    #Check if the step the agent takes is valid (reachable and within max steps).
 def check_valid_step(chess_board, adv_pos, start_pos, end_pos, barrier_dir, max_step):
    # Endpoint already has barrier or is boarder
    r, c = end_pos
    if chess_board[r, c, barrier_dir]:
        return False
    if np.array_equal(start_pos, end_pos):
        return True

    # BFS
    state_queue = [(start_pos, 0)]
    visited = {tuple(start_pos)}
    is_reached = False
    moves = ((-1, 0), (0, 1), (1, 0), (0, -1))
    while state_queue and not is_reached:
        cur_pos, cur_step = state_queue.pop(0)
        r, c = cur_pos
        if cur_step == max_step:
            break
        for dir, move in enumerate(moves):
            if chess_board[r, c, dir]:
                continue

            next_pos = cur_pos + move
            if np.array_equal(next_pos, adv_pos) or tuple(next_pos) in visited:
                continue
            if np.array_equal(next_pos, end_pos):
                is_reached = True
                break

            visited.add(tuple(next_pos))
            state_queue.append((next_pos, cur_step + 1))

    return is_reached

def set_barrier(chessboard, r, c, dir):
    # Set the barrier to True
    opposites = {0: 2, 1: 3, 2: 0, 3: 1}
    moves = ((-1, 0), (0, 1), (1, 0), (0, -1))
    chessboard[r, c, dir] = True
    # Set the opposite barrier to True
    move = moves[dir]
    chessboard[r + move[0], c + move[1], opposites[dir]] = True

    return chessboard


def check_endgame(chessboard, board_size, my_pos, adv_pos):
    """
    Check if the game ends and compute the current score of the agents.

    Returns
    -------
    is_endgame : bool
        Whether the game ends.
    player_1_score : int
        The score of player 1.
    player_2_score : int
        The score of player 2.
    """
    # Union-Find
    moves = ((-1, 0), (0, 1), (1, 0), (0, -1))
    father = dict()
    for r in range(board_size):
        for c in range(board_size):
            father[(r, c)] = (r, c)

    def find(pos):
        if father[pos] != pos:
            father[pos] = find(father[pos])
        return father[pos]

    def union(pos1, pos2):
        father[pos1] = pos2

    for r in range(board_size):
        for c in range(board_size):
            for dir, move in enumerate(
                    moves[1:3]
            ):  # Only check down and right
                if chessboard[r, c, dir + 1]:
                    continue
                pos_a = find((r, c))
                pos_b = find((r + move[0], c + move[1]))
                if pos_a != pos_b:
                    union(pos_a, pos_b)

    for r in range(board_size):
        for c in range(board_size):
            find((r, c))
    p0_r = find(tuple(my_pos))
    p1_r = find(tuple(adv_pos))
    p0_score = list(father.values()).count(p0_r)
    p1_score = list(father.values()).count(p1_r)
    if p0_r == p1_r:
        return False, p0_score, p1_score
    player_win = None
    win_blocks = -1
    if p0_score > p1_score:
        player_win = 0
        win_blocks = p0_score
    elif p0_score < p1_score:
        player_win = 1
        win_blocks = p1_score
    else:
        player_win = -1  # Tie

    return True, p0_score, p1_score