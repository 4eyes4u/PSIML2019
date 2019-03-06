import sys
import os
import io
import numpy as np
import time
from itertools import product as iter_product

np.set_printoptions(threshold=np.inf)

def parse_tetris_data(data):
    board = np.zeros(shape=(H, W), dtype=np.int32)
    for i, j in iter_product(range(H), range(W)):
        board[i][j] = 1 if data[i][j] == '#' else 0
    
    zero_len = 0
    figures = []; block = []
    for row in range(H, len(data)):
        cur_len = len(data[row])
        zero_len = zero_len + (1 if cur_len == 0 else 0)
        if zero_len == 2:
            figures.append(np.asarray(block, dtype=np.int32))
            block = []
            zero_len = 1
        elif cur_len != 0:
            line = [1 if entry == '#' else 0 for entry in data[row]]
            block.append(line)

    return board, figures

def rotate_single_figure(figure):
    rot_0 = np.copy(figure)
    rot_90 = np.flip(rot_0.T, axis=0)
    rot_180 = np.flip(rot_90.T, axis=0)
    rot_270 = np.flip(rot_180.T, axis=0)
    return [rot_0, rot_90, rot_180, rot_270]

def rotate_figures(figures):
    figures_rotated = []
    for figure in figures:
        figures_rotated += rotate_single_figure(figure)
    return figures_rotated

class FirstSubtask:
    def __init__(self, board, figures):
        self.board = np.copy(board)
        self.figures = figures.copy()
        self.n_figures = len(figures)

    def _calc_score(self, row, col, figure):
        h, w = figure.shape
        # putting figure on board
        self.board[row:row + h, col:col + w] = np.bitwise_xor(self.board[row:row + h, col:col + w], figure)
        # calculate how many rows are fully occuppied
        score = np.asarray([np.sum(self.board[i]) == W for i in range(row, row + h)], dtype=np.int32)
        score = np.sum(score)
        # removing figure
        self.board[row:row + h, col:col + w] = np.bitwise_xor(self.board[row:row + h, col:col + w], figure)
        return score
    
    def _valid_move(self, row, col, figure):
        # elementwise product
        h, w = figure.shape
        match = np.sum(np.multiply(self.board[row:row + h, col:col + w], figure))
        return match == 0

    def single_shape(self, figure):
        h, w = figure.shape
        scores = np.zeros(shape=(W - w + 1, ), dtype=np.int32)
        for col in range(W - w + 1):
            for row in range(H - h + 1):
                # shape can't be placed here; previous move was last
                if not self._valid_move(row, col, figure):
                    scores[col] = self._calc_score(row - 1, col, figure)
                    break

        start_col = np.argmax(scores, axis=0)
        return scores[start_col], start_col

    def solve(self):
        scores = np.zeros((self.n_figures, 2), np.int32)
        for i, figure in enumerate(self.figures):
            scores[i] = np.asarray([self.single_shape(figure)], dtype=np.int32)
        figure_id = np.argmax(scores, axis=0)[0]
        print('{} {} {}'.format(figure_id // 4, figure_classes[figure_id % 4], scores[figure_id][1]))

class SecondSubtask:
    def __init__(self, board, figures):
        self.board = np.copy(board)
        self.figures = figures
        self.n_figures = len(figures)
        self.best_figure_id = -1
        self.mark = np.full(shape=(H, W), fill_value=-1, dtype=np.int32)
        self.best_score = -1
        self.end_row = -1
        self.end_col = -1

    def _calc_score(self, row, col, figure):
        h, w = figure.shape
        if row + h - 1 >= H or col + w - 1 >= W or col < 0:
            return -1
        # putting figure on board
        self.board[row:row + h, col:col + w] = np.bitwise_xor(self.board[row:row + h, col:col + w], figure)
        # calculate how many rows are fully occuppied
        score = np.asarray([np.sum(self.board[i]) == W for i in range(row, row + h)], dtype=np.int32)
        score = np.sum(score)
        # removing figure
        self.board[row:row + h, col:col + w] = np.bitwise_xor(self.board[row:row + h, col:col + w], figure)
        return score

    def _valid_move(self, row, col, figure):
        # elementwise product
        h, w = figure.shape
        if row + h - 1 >= H or col + w - 1 >= W or col < 0:
            return False
        # matching between board and figure should not exist
        match = np.sum(np.multiply(self.board[row:row + h, col:col + w], figure))
        return match == 0

    def _dfs(self, row, col, figure_id):
        figure = self.figures[figure_id]
        h, w = figure.shape
        if not self._valid_move(row + 1, col, figure):
            cur_score = self._calc_score(row, col, figure)
            if cur_score > self.best_score:
                self.best_figure_id = figure_id
                self.best_score = cur_score
                self.end_row = row
                self.end_col = col
        else:
            for step in range(0, W):
                if self._valid_move(row + 1, col + step, figure) and self.mark[row + 1][col + step] == -1:
                    self.mark[row + 1][col + step] = row * H + col
                    self._dfs(row + 1, col + step, figure_id)
                else: break
            
            for step in range(-1, -W, -1):
                if self._valid_move(row + 1, col + step, figure) and self.mark[row + 1][col + step] == -1:
                    self.mark[row + 1][col + step] = row * H + col
                    self._dfs(row + 1, col + step, figure_id)
                else: break
    
    def _make_path(self, figure_id):
        self.mark = np.full(shape=(H, W), fill_value=-1, dtype=np.int32)
        for start_col in range(W):
            self._dfs(0, start_col, figure_id)

        path = []
        cur_row = self.end_row
        cur_col = self.end_col
        while cur_row != 0:
            prev = self.mark[cur_row][cur_col]
            prev_row = prev // H
            prev_col = prev % H
            path.append(cur_col - prev_col)
            cur_row = prev_row
            cur_col = prev_col
        
        path = path[ : : -1]
        print(figure_id // 4, figure_classes[figure_id % 4], cur_col, sep=' ', end=' ')
        print(*path, sep=' ')

    def solve(self):
        for i in range(self.n_figures):
            self.mark = np.full(shape=(H, W), fill_value=-1, dtype=np.int32)
            for col in range(W):
                self._dfs(0, col, i)
        
        self._make_path(self.best_figure_id)

if __name__ == "__main__":
    # constants
    H = 20; W = 10
    figure_classes = ['0', '90', '180', '270']
    
    tetris_path = input()
    with open(tetris_path, 'r') as f:
        data = f.read()
        data = ''.join(data)
        data = data.split('\n')
    
    board, figures = parse_tetris_data(data)
    figures_rotated = rotate_figures(figures)

    fs_solver = FirstSubtask(board, figures_rotated)
    fs_solver.solve()

    ss_solver = SecondSubtask(board, figures_rotated)
    ss_solver.solve()