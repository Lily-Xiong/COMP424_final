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
        # dummy return
        return my_pos, self.dir_map["u"]

    def random_move(self, chess_board, my_pos, adv_pos, max_step):
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



class MonteCarloSearchTree():
    def __init__(self, move, parentNode):
        self.rootNode = aNode

    def expand(self, node):
        # apply default policy, if node is not a terminal node, then expand by one node
        # dummy return
        return node

    def simulate(self, node):
        # simulate game from node
        return node

    def backpropagation(self):
        # backpropagate
        # dummy return
        return


class TreeNode:
    def __init__(self, move, chessboard, parentNode=None):
        self.parent = parentNode
        self.children = []
        self.num_of_visit = 0
        self.num_of_wins = 0
        self.move = move
        self.chessboard = chessboard


    # select the best child node using UCT
    def find_best_child_node_by_uct(self):
        #a function to calculate UCT Value
        max_child = self.get_max_uct_children()
        return max_child

    def get_max_uct_children(self):

        # array that stores UCT value for each child of the current node
        UCT_arr = np.zeros(len(self.children))

        # iterate through the list of children and caluculate the UCT for each
        for i in range(len(self.children)):
            UCT_arr[i] = self.children[i].uct

        return self.children[max(UCT_arr)]

    def uct(self):
        return (self.num_of_wins / self.num_of_visit) + math.sqrt(2) * (math.sqrt(
            math.log(self.parent.num_of_visit) / self.num_of_visit))

    def update_data(self, game_result):
        self.num_of_visit += 1
        if game_result == "win":
            self.num_of_wins += 1

    def is_terminal(self):
        if len(self.children) == 0:
            return True
        return False

    def select_best_node(self):
        bestNode = self

        while not self.is_terminal():
            bestNode = self.find_best_child_node_by_uct(bestNode)

        return bestNode

    def expandNode(self, chess_board, max_step, adv_pos):
        parent_node = self
        new_pos = getNextPossibleMove(chess_board, self.pos, adv_pos, )
        new_node = TreeNode(parent_node, )

    def simulation(self, chess_board, max_step):
        # returns the final board after the game has ended

        return end_state

    def get_next_possible_move(chess_board, my_pos, adv_pos, dir, max_step):
        new_pos = generate_random_move(my_pos, max_step)
        while check_valid_step(chess_board, adv_pos, my_pos, new_pos, dir, max_step) == False:
            new_pos = generate_random_move(my_pos, max_step)

        return new_pos

    def generate_random_move(self, max_step):
        x, y = self.pos
        random_x = random.randint(x - max_step, x + max_step)
        random_y = random.randint(x - max_step, x + max_step)
        random_dir = random.randint(0, 3)

        return ((random_x, random_y), random_dir)



    def check_valid_step(chess_board, adv_pos, start_pos, end_pos, barrier_dir, max_step):
        """
        Check if the step the agent takes is valid (reachable and within max steps).

        Parameters
        ----------
        start_pos : tuple
            The start position of the agent.
        end_pos : np.ndarray
            The end position of the agent.
        barrier_dir : int
            The direction of the barrier.
        """
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

    def check_endgame(self, board_size, my_pos, adv_pos):
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
                    if self.chessboard[r, c, dir + 1]:
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


class BoardState:
    def __init__(self, chess_board, my_pos, adv_pos, max_step):
        self.board = chess_board
        self.my_pos = my_pos
        self.adv_pos = adv_pos
        self.max_step = max_step
