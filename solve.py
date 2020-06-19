"""
Author: Mohamed Afify
Date: May 2020

Before reading:
    The solving algorithm is divided into three separate methods, where
    each method can only solve the sudoku to a certain extent (but not entirely), unless
    the sudoku is fairly easy:

        1. To solve a sudoku, each number must satisfy the row, column, and block rules

    In Method 1:
        1. We attempt to solve the sudoku using (block, row, column) notation for each cell.
        2. We iterate over Blocks, and blocks are numbered as shown:
                    0   1   2
                    3   4   5
                    6   7   8
        3. Each cell is numbered according to its (row, col) location in this block, so, for
            any block, cells are numbered as shown:
                    (0,0)   (0,1)   (0,2)
                    (1,0)   (1,1)   (1,2)
                    (2,0)   (2,2)   (2,2)
        4. We use a note-taking method to have a better solver

    In Method 2:
        1. We attempt to solve the sudoku using (row, column) notation for each cell.
        2. We iterate over rows, and rows are numbered from top to bottom (0 to 9)

    In Method 3:
        1. Same as Method 2, but we iterate over columns
        2. Columns are numbered from left to right (0 to 9)

    In Method 4:
        1. We solve on a cell level.
        2. For example, if there are only 3 free cells in a row, sometimes the column
            rule can give a hint if there's a free cell that can accept only one number
            of the three missing numbers (only that number can exist in that cell, the other
            two are discarded using the column rule)
"""

import sudoku_solver as sd


def get_column(matrix, i):
    return [row[i] for row in matrix]


def solve_grid(ord_grid, n):
    """
    Arguments:
    ord_grid -- the unsolved sudoku grid
    n -- the number of iterations, better be high for hard sudokus

    Returns:
    Sudoku.ordinary_grid -- the solved (hopefully) sudoku grid
    zeros_indices -- A list of coordinates of the  originally
                     empty sudoku cells (before solving)
    """

    # Construct a grid of blocks where each row is a block
    blk_grid = [[ord_grid[0][0:3], ord_grid[1][0:3], ord_grid[2][0:3]],
                [ord_grid[0][3:6], ord_grid[1][3:6], ord_grid[2][3:6]],
                [ord_grid[0][6:9], ord_grid[1][6:9], ord_grid[2][6:9]],
                [ord_grid[3][0:3], ord_grid[4][0:3], ord_grid[5][0:3]],
                [ord_grid[3][3:6], ord_grid[4][3:6], ord_grid[5][3:6]],
                [ord_grid[3][6:9], ord_grid[4][6:9], ord_grid[5][6:9]],
                [ord_grid[6][0:3], ord_grid[7][0:3], ord_grid[8][0:3]],
                [ord_grid[6][3:6], ord_grid[7][3:6], ord_grid[8][3:6]],
                [ord_grid[6][6:9], ord_grid[7][6:9], ord_grid[8][6:9]]]

    # Instantiate a Sudoku object
    Sudoku = sd.Sudoku(grid=blk_grid, ordinaryGrid=ord_grid)

    for i in range(1, n):                                               # Repeat the algorithm n times
        for number in range(1, 10):                                     # Repeat the algorithm for all numbers
            # 1. Solving using blocks
            found_instances = Sudoku.find_num_blk_instances(number)     # a list of [block, row, column] per instance
            valid_blocks = Sudoku.find_valid_blocks_for_num(found_instances)    # a list of possible blocks
            for block in valid_blocks:
                valid_cells = Sudoku.find_valid_cells_in_blk_for_num(number, block, found_instances)
                Sudoku.add_note_or_num(number, block, valid_cells)      # Add a number (solve) or add a note

            # 2. Solving using rows
            ord_found_instances = Sudoku.find_ord_instances_for_num(number)     # a list of [row, column] per instance
            valid_rows = Sudoku.find_valid_rows_for_num(ord_found_instances)    # a list of possible rows
            for row in valid_rows:
                row_valid_cells = Sudoku.find_valid_cells_in_row(row, ord_found_instances)
                Sudoku.add_to_row(number, row, row_valid_cells)

            # 3. Solving using columns
            ord_found_instances = Sudoku.find_ord_instances_for_num(number)     # a list of [row, column] per instance
            valid_cols = Sudoku.find_valid_cols_for_num(ord_found_instances)    # a list of possible columns
            for col in valid_cols:
                col_valid_cells = Sudoku.find_valid_cells_in_col(col, ord_found_instances)
                Sudoku.add_to_col(number, col, col_valid_cells)

        if Sudoku.solved:
            return Sudoku.ordinaryGrid, Sudoku.emptyCells

        # 4. Solving using cells
        for row in range(0, 9):
            Sudoku.solve_by_cell_in_row(row)
        for col in range(0, 9):
            Sudoku.solve_by_cell_in_col(col)

        if Sudoku.solved:
            return Sudoku.ordinaryGrid, Sudoku.emptyCells

    return Sudoku.ordinaryGrid, Sudoku.emptyCells

