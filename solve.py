import sudoku_solver as sd


def get_column(matrix, i):
    return [row[i] for row in matrix]

#
# hard_1 = [
#     [0, 1, 0, 0, 0, 0, 7, 0, 3],
#     [7, 0, 0, 8, 1, 0, 0, 2, 0],
#     [2, 0, 0, 0, 3, 0, 0, 9, 0],
#     [0, 0, 4, 0, 0, 0, 0, 0, 5],
#     [0, 0, 0, 1, 7, 2, 0, 0, 0],
#     [6, 0, 0, 0, 0, 0, 1, 0, 0],
#     [0, 6, 0, 0, 5, 0, 0, 0, 8],
#     [0, 2, 0, 0, 4, 8, 0, 0, 7],
#     [9, 0, 7, 0, 0, 0, 0, 4, 0]
# ]
#
#
# ord_grid = hard_1


def get_zeros(to_draw):
    """
    Arguments:
    to_draw -- A (9 x 9) array (sudoku grid to be shown)

    Returns:
    list_z -- A list of coordinates of empty (blank) sudoku cells
    """

    list_z = []
    for row in range(0, 9):
        for col in range(0, 9):
            if to_draw[row][col] == 0:
                list_z.append([row, col])
    return list_z


def solve_grid(ord_grid):
    """
    Arguments:
    ord_grid -- the unsolved sudoku grid

    Returns:
    Sudoku.ordinary_grid -- the solved (hopefully) sudoku grid
    zeros_indices -- A list of coordinates of the  originally
                     empty sudoku cells (before solving)
    """

    blk_grid = [[ord_grid[0][0:3], ord_grid[1][0:3], ord_grid[2][0:3]],
                [ord_grid[0][3:6], ord_grid[1][3:6], ord_grid[2][3:6]],
                [ord_grid[0][6:9], ord_grid[1][6:9], ord_grid[2][6:9]],
                [ord_grid[3][0:3], ord_grid[4][0:3], ord_grid[5][0:3]],
                [ord_grid[3][3:6], ord_grid[4][3:6], ord_grid[5][3:6]],
                [ord_grid[3][6:9], ord_grid[4][6:9], ord_grid[5][6:9]],
                [ord_grid[6][0:3], ord_grid[7][0:3], ord_grid[8][0:3]],
                [ord_grid[6][3:6], ord_grid[7][3:6], ord_grid[8][3:6]],
                [ord_grid[6][6:9], ord_grid[7][6:9], ord_grid[8][6:9]]]
    Sudoku = sd.Sudoku(grid=blk_grid, ordinary_grid=ord_grid)
    zero_indices = get_zeros(Sudoku.ordinary_grid)

    for i in range(1, 100):
        for number in range(1, 10):
            instancesOfNumber = Sudoku.find_num_blk_instances(number)  # [block, row, column] for the entire grid
            valid_blocks = Sudoku.find_valid_blocks_for_num(instancesOfNumber)

            for block in valid_blocks:
                valid_cells = Sudoku.find_valid_cells_in_blk_for_num(number, block, instancesOfNumber)
                Sudoku.add_note_or_num(number, block, valid_cells)

            ord_instancesOfNumber = Sudoku.find_ord_instances_for_num(number)
            valid_rows = Sudoku.find_valid_rows_for_num(ord_instancesOfNumber)
            for row in valid_rows:
                row_valid_cells = Sudoku.find_valid_cells_in_row(row, ord_instancesOfNumber)
                Sudoku.add_to_row(number, row, row_valid_cells)

            ord_instancesOfNumber = Sudoku.find_ord_instances_for_num(number)
            valid_cols = Sudoku.find_valid_cols_for_num(ord_instancesOfNumber)
            for col in valid_cols:
                col_valid_cells = Sudoku.find_valid_cells_in_col(col, ord_instancesOfNumber)
                Sudoku.add_to_col(number, col, col_valid_cells)

        for row in range(0, 9):
            Sudoku.solve_by_cell_in_row(row)
        for col in range(0, 9):
            Sudoku.solve_by_cell_in_col(col)

        if len(zero_indices) == Sudoku.added:
            print("😂😂😂😂Done😂😂😂😂")
            break

    return Sudoku.ordinary_grid, zero_indices

    # print(Sudoku.added)
    # # print(Sudoku.notes)
    # ds.draw_sudoku(Sudoku.ordinary_grid, zero_indices, save=True)
