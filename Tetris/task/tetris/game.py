# Write your code here
import numpy as np
from collections import deque


class Matrix:

    def __init__(self, x: int, y: int):
        self.deque = deque()
        self.matrix = None
        self.x = x
        self.y = y

    def set_proper_symbol(self, symbol: int) -> str:
        return '0' if symbol == 1 else '-'

    def __str__(self):
        string = ''
        for row in self.matrix:
            string += " ".join(map(self.set_proper_symbol, row)) + '\n'
        return string

    def create_empty_matrix(self):
        return np.zeros([self.y, self.x], dtype=int)


class GameBoard(Matrix):

    def __init__(self, x: int, y: int):
        super().__init__(x, y)
        self.matrix = self.create_empty_matrix()
        self.temp_matrix = self.create_empty_matrix()
        self.matrix_left_border = self.create_empty_matrix()
        self.matrix_right_border = self.create_empty_matrix()
        self.matrix_top_border = self.create_empty_matrix()
        self.matrix_bottom_border = self.create_empty_matrix()
        self.block = None
        self.endgame = False

        for i in range(y):
            self.matrix_left_border[i][0] = 1
            self.matrix_right_border[i][self.x - 1] = 1
        for i in range(x):
            self.matrix_top_border[0][i] = 1
            self.matrix_bottom_border[self.y - 1][i] = 1

    def is_collision(self, matrix: "np array", move_left: bool = False, move_right: bool = False, move_down: bool = False,
                     initial: bool = False) -> bool:
        if np.any(np.logical_and(self.matrix, matrix)):
            if self.block.initial:
                self.endgame = True
            return True
        if move_down and not initial and np.any(np.logical_and(self.matrix_top_border, matrix)):
            return True
        if move_left and np.any(np.logical_and(self.matrix_right_border, matrix)):
            return True
        if move_right and np.any(np.logical_and(self.matrix_left_border, matrix)):
            return True
        return False

    def position_block(self, matrix: "np array", block: "np array", x: int, y: int, initial = False) -> "np array":
        matrix[y:y + block.matrix.shape[0], x:x + block.matrix.shape[1]] += block.matrix
        return not self.is_collision(matrix, move_down=True, move_right=True, move_left=True, initial=initial)

    def create_new_block(self, block: "np array") -> bool:
        if self.block is not None:
            return False
        self.block = block
        matrix = self.create_empty_matrix()
        x = (self.x - self.block.x) // 2
        y = 0
        if self.position_block(matrix, self.block, x, y, True):
            self.temp_matrix = matrix
            block.x_pos = x
            block.y_pos = y
            return True
        return False

    def roll_left(self) -> bool:
        if self.block is None:
            return False
        matrix = np.roll(self.temp_matrix, -1, axis=1)
        if not self.is_collision(matrix, move_left=True):
            self.temp_matrix = matrix
            self.block.x_pos -= 1
            return True
        return False

    def roll_right(self) -> bool:
        if self.block is None:
            return False
        matrix = np.roll(self.temp_matrix, 1, axis=1)
        if not self.is_collision(matrix, move_right=True):
            self.temp_matrix = matrix
            self.block.x_pos += 1
            return True
        return False

    def roll_down(self) -> bool:
        if self.block is None:
            return False
        matrix = np.roll(self.temp_matrix, 1, axis=0)
        if not self.is_collision(matrix, move_down=True):
            self.temp_matrix = matrix
            self.block.y_pos += 1
            return True
        else:
            self.matrix = np.logical_or(self.matrix, self.temp_matrix)
            self.block = None
            self.temp_matrix = self.create_empty_matrix()
        return False

    def rotate(self) -> bool:
        self.block.rotate_counter_clockwise()
        matrix = self.create_empty_matrix()
        if self.position_block(matrix, self.block, self.block.x_pos, self.block.y_pos, self.block.initial):
            self.temp_matrix = matrix
            return True
        self.block.rotate_counter_clockwise()
        return False

    def __str__(self):
        string = ''
        matrix = self.temp_matrix + self.matrix
        for row in matrix:
            string += " ".join(map(self.set_proper_symbol, row)) + '\n'
        return string

    def delete_full_row(self):
        full_rows = np.all(self.matrix, axis=1)
        self.matrix = np.delete(self.matrix, np.where(full_rows), axis=0)
        self.matrix = np.concatenate([np.zeros([np.sum(full_rows), self.x], dtype=int), self.matrix], axis=0)


class Block(Matrix):

    def __init__(self, x: int, y: int):
        super().__init__(x, y)
        self.deque = deque()
        self.x_pos = 0
        self.y_pos = 0

    @property
    def initial(self):
        return self.y_pos == 0

    def rotate_clockwise(self):
        self.deque.rotate(1)
        self.matrix = self.deque[0]

    def rotate_counter_clockwise(self):
        self.deque.rotate(-1)
        self.matrix = self.deque[0]

    def create_initial_shapes(self, initial_shape: "list with shapes"):
        for i in range(len(initial_shape)):
            matrix = self.create_empty_matrix()
            for j in range(len(initial_shape[i])):
                element = initial_shape[i][j]
                matrix[element // self.x][element % self.x] = '1'
            self.deque.append(matrix)
        self.matrix = self.deque[0]


class EmptyBlock(Block):
    def __init__(self):
        super().__init__(10, 4)
        self.create_initial_shapes([[]])


class IBlock(Block):
    def __init__(self):
        super().__init__(10, 4)
        self.create_initial_shapes([[4, 14, 24, 34], [3, 4, 5, 6]])


class SBlock(Block):
    def __init__(self):
        super().__init__(10, 4)
        self.create_initial_shapes([[5, 4, 14, 13], [4, 14, 15, 25]])


class ZBlock(Block):
    def __init__(self):
        super().__init__(10, 4)
        self.create_initial_shapes([[4, 5, 15, 16], [5, 15, 14, 24]])


class LBlock(Block):
    def __init__(self):
        super().__init__(10, 4)
        self.create_initial_shapes([[4, 14, 24, 25], [5, 15, 14, 13], [4, 5, 15, 25], [6, 5, 4, 14]])


class JBlock(Block):
    def __init__(self):
        super().__init__(10, 4)
        self.create_initial_shapes([[5, 15, 25, 24], [15, 5, 4, 3], [5, 4, 14, 24], [4, 14, 15, 16]])


class OBlock(Block):
    def __init__(self):
        super().__init__(10, 4)
        self.create_initial_shapes([[4, 14, 15, 5]])


class TBlock(Block):
    def __init__(self):
        super().__init__(10, 4)
        self.create_initial_shapes([[4, 14, 24, 15], [4, 13, 14, 15], [5, 15, 25, 14], [4, 5, 6, 15]])


def get_new_block():
    cmd = input()
    b = None
    match cmd:
        case 'O':
            b = OBlock()
        case 'I':
            b = IBlock()
        case 'S':
            b = SBlock()
        case 'Z':
            b = ZBlock()
        case 'L':
            b = LBlock()
        case 'J':
            b = JBlock()
        case 'T':
            b = TBlock()
    return b



x, y = map(int, input().split())
board = GameBoard(x, y)
print(board)

while not board.endgame:
    cmd = input()
    match cmd:
        case "rotate":
            board.rotate()
            board.roll_down()
        case "left":
            board.roll_left()
            board.roll_down()
        case "right":
            board.roll_right()
            board.roll_down()
        case "down":
            board.roll_down()
        case "exit":
            break
        case "piece":
            b = get_new_block()
            board.create_new_block(b)
        case "break":
            board.delete_full_row()
        case _:
            continue
    print(board)

print("Game Over!")
