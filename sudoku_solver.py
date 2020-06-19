"""
Author: Mohamed Afify
Date: May 2020

Overview:
    - The Sudoku class contains all the information about the sudoku and the methods
    needed for the algorithm
    - It would be hard to explain everything in this file, but it shouldn't be hard
    if you have read and understood the solve.py file

"""


def get_column(matrix, i):
    return [row[i] for row in matrix]


class Sudoku:
    def __init__(self, grid, ordinaryGrid):
        self.grid = grid
        self.ordinaryGrid = ordinaryGrid
        self.notes = []
        self.added = 0
        self.solved = False

        empty_cells = []
        for row in range(0, 9):
            for col in range(0, 9):
                if self.ordinaryGrid[row][col] == 0:
                    empty_cells.append([row, col])
        self.emptyCells = empty_cells

    def __repr__(self):
        print(self.grid)
        print(self.notes)
        print(self.added)

    ##########################################################################################
    # ------------------------------Adding Numbers and Updating Blocks------------------------
    ##########################################################################################
    def add_num_in_blk(self, number, block, row, column):
        self.grid[block][row][column] = number
        self.added += 1
        self.update_ord_grid(self.grid)
        if len(self.emptyCells) == self.added:
            self.solved = True

    def add_num_in_ord(self, number, row, column):
        self.ordinaryGrid[row][column] = number
        self.added += 1
        self.update_blk_grid(self.ordinaryGrid)
        if len(self.emptyCells) == self.added:
            self.solved = True

    def update_ord_grid(self, blk_grid):
        # Updates the block grid using the ordinary grid
        self.ordinaryGrid = [blk_grid[0][0] + blk_grid[1][0] + blk_grid[2][0],
                              blk_grid[0][1] + blk_grid[1][1] + blk_grid[2][1],
                              blk_grid[0][2] + blk_grid[1][2] + blk_grid[2][2],
                              blk_grid[3][0] + blk_grid[4][0] + blk_grid[5][0],
                              blk_grid[3][1] + blk_grid[4][1] + blk_grid[5][1],
                              blk_grid[3][2] + blk_grid[4][2] + blk_grid[5][2],
                              blk_grid[6][0] + blk_grid[7][0] + blk_grid[8][0],
                              blk_grid[6][1] + blk_grid[7][1] + blk_grid[8][1],
                              blk_grid[6][2] + blk_grid[7][2] + blk_grid[8][2]]

    def update_blk_grid(self, ord_grid):
        # Updates the ordinary grid using the block grid
        self.grid = [[ord_grid[0][0:3], ord_grid[1][0:3], ord_grid[2][0:3]],
                     [ord_grid[0][3:6], ord_grid[1][3:6], ord_grid[2][3:6]],
                     [ord_grid[0][6:9], ord_grid[1][6:9], ord_grid[2][6:9]],
                     [ord_grid[3][0:3], ord_grid[4][0:3], ord_grid[5][0:3]],
                     [ord_grid[3][3:6], ord_grid[4][3:6], ord_grid[5][3:6]],
                     [ord_grid[3][6:9], ord_grid[4][6:9], ord_grid[5][6:9]],
                     [ord_grid[6][0:3], ord_grid[7][0:3], ord_grid[8][0:3]],
                     [ord_grid[6][3:6], ord_grid[7][3:6], ord_grid[8][3:6]],
                     [ord_grid[6][6:9], ord_grid[7][6:9], ord_grid[8][6:9]]]

    ##########################################################################################
    # ------------------------------By block Solving------------------------
    ##########################################################################################
    def find_num_blk_instances(self, number):
        locations = []
        for block in range(0, 9):
            for row in range(0, 3):
                for column in range(0, 3):
                    if self.grid[block][row][column] == number:
                        locations.append([block, row, column])
        return locations

    @staticmethod
    def find_valid_blocks_for_num(instances):
        invalid_blocks = []
        valid_blocks = []
        for location in instances:
            invalid_blocks.append(location[0])
        for block in range(0, 9):
            if block not in invalid_blocks:
                valid_blocks.append(block)
        return valid_blocks

    def find_free_cells_in_blk(self, block):  # returns [row, column] for free cells in the current block
        free_cells = []
        for row in range(0, 3):
            for column in range(0, 3):
                if self.grid[block][row][column] == 0:
                    free_cells.append([row, column])
        return free_cells

    def find_invalid_rows_in_blk_for_num(self, number, block, instances):
        # Returns 0 or 1 or 2 or a combination
        rows = []
        def_block = block
        if block in [0, 1, 2]:
            block = 0
        elif block in [3, 4, 5]:
            block = 3
        elif block in [6, 7, 8]:
            block = 6

        row_blocks = [block, block + 1, block + 2]
        for instance in instances:
            if instance[0] in row_blocks:
                rows.append(instance[1])

        # Taking Row Notes into account
        row_blocks.remove(def_block)
        for i in range(len(self.notes) - 1):
            if [self.notes[i][0], self.notes[i][1]] == [number, row_blocks[0]] or [self.notes[i][0],
                                                                                   self.notes[i][1]] == [number,
                                                                                                         row_blocks[1]]:
                if self.notes[i][0] == self.notes[i + 1][0] and self.notes[i][1] and self.notes[i + 1][1] and \
                        self.notes[i][2] == self.notes[i + 1][2] and self.notes[i][3] not in rows:
                    rows.append(self.notes[i][2])
                    break
        return rows

    def find_invalid_cols_in_blk_for_num(self, number, block, instances):
        columns = []
        def_block = block
        if block in [0, 3, 6]:
            block = 0
        elif block in [1, 4, 7]:
            block = 1
        elif block in [2, 5, 8]:
            block = 2

        column_blocks = [block, block + 3, block + 6]
        for instance in instances:
            if instance[0] in column_blocks:
                columns.append(instance[2])

        # Taking Column Notes into account
        column_blocks.remove(def_block)
        for i in range(len(self.notes) - 1):
            if [self.notes[i][0], self.notes[i][1]] == [number, column_blocks[0]] or \
                    [self.notes[i][0], self.notes[i][1]] == [number, column_blocks[1]]:

                if self.notes[i][0] == self.notes[i + 1][0] and self.notes[i][1] and self.notes[i + 1][1] and \
                        self.notes[i][3] == self.notes[i + 1][3] and self.notes[i][3] not in columns:
                    columns.append(self.notes[i][3])
                    break

        return columns

    def find_valid_cells_in_blk_for_num(self, number, block, instances):
        invalid_rows = self.find_invalid_rows_in_blk_for_num(number, block, instances)
        invalid_cols = self.find_invalid_cols_in_blk_for_num(number, block, instances)
        free_cells = self.find_free_cells_in_blk(block)
        valid_cells = []

        for row in range(3):
            for col in range(3):
                if row not in invalid_rows and col not in invalid_cols and [row, col] in free_cells:
                    valid_cells.append([row, col])

        return valid_cells

    def add_note_or_num(self, number, block, valid_cells):
        # Adds a note or inserts a missing number
        # Adds a note if the number of valid cells in this block for a certain number is 2
        # Inserts a number otherwise, and also removes notes related to that number in self.notes
        if len(valid_cells) == 1:
            for note in self.notes:
                if [note[0], note[1]] == [number, block]:
                    self.notes.remove(note)
            # repeat the operation because notes are added in twos
            for note in self.notes:
                if [note[0], note[1]] == [number, block]:
                    self.notes.remove(note)
            self.add_num_in_blk(number, block, valid_cells[0][0], valid_cells[0][1])

        # Only adds a note if the number of valid cells is two, and also if both notes have
        # the same row or the same column.
        elif len(valid_cells) == 2:
            note1 = [number, block, valid_cells[0][0], valid_cells[0][1]]
            note2 = [number, block, valid_cells[1][0], valid_cells[1][1]]
            if note1 not in self.notes and note2 not in self.notes and (
                    valid_cells[0][0] == valid_cells[1][0] or valid_cells[0][1] == valid_cells[1][1]):
                self.notes.append([number, block, valid_cells[0][0], valid_cells[0][1]])
                self.notes.append([number, block, valid_cells[1][0], valid_cells[1][1]])

    ##########################################################################################
    # ------------------------------Solving using rows and columns (Simple)-------------------
    ##########################################################################################
    def solve_by_row(self):
        # Solving the sudoku using the row rule
        # By looking for rows having a single empty slot (0) and replacing it with the missing number
        # input: ordinary grid (9x9)   >>>   output: inserting missing number
        for row in range(0, 9):
            numbers_in_row = self.ordinaryGrid[row]
            if numbers_in_row.count(0) == 1:
                k = numbers_in_row.index(0)
                for number in range(1, 10):
                    if number not in numbers_in_row:
                        self.add_num_in_ord(number, row, k)

    def solve_by_col(self):
        # Solving the sudoku using the column rule
        # By looking for columns having a single empty slot (0) and replacing it with the missing number
        # input: ordinary grid (9x9)   >>>   output: inserting missing number
        ord_grid_t = [list(x) for x in zip(*self.ordinaryGrid)]  # Getting the transpose to make the algorithm easier
        for col in range(0, 9):
            numbers_in_col = ord_grid_t[col]
            if numbers_in_col.count(0) == 1:
                for number in range(1, 10):
                    if number not in numbers_in_col:
                        self.add_num_in_ord(number, numbers_in_col.index(0), col)

    ##########################################################################################
    # ------------------------------Solving by rows and columns (complex)---------------------
    ##########################################################################################
    def find_ord_instances_for_num(self, number):
        locations = []
        for row in range(0, 9):
            for col in range(0, 9):
                if self.ordinaryGrid[row][col] == number:
                    locations.append([row, col])
        return locations

    @staticmethod
    def find_valid_rows_for_num(ord_instances):
        invalid_rows = []
        valid_rows = []

        for location in ord_instances:
            invalid_rows.append(location[0])

        for row in range(0, 9):
            if row not in invalid_rows:
                valid_rows.append(row)

        return valid_rows

    @staticmethod
    def find_valid_cols_for_num(ord_instances):
        invalid_cols = []
        valid_cols = []

        for location in ord_instances:
            invalid_cols.append(location[1])

        for col in range(0, 9):
            if col not in invalid_cols:
                valid_cols.append(col)

        return valid_cols

    def find_free_cells_in_row(self, row):
        row_free_cells = []
        for col in range(0, 9):
            if self.ordinaryGrid[row][col] == 0:
                row_free_cells.append(col)
        return row_free_cells

    def find_free_cells_in_col(self, col):
        col_free_cells = []
        for row in range(0, 9):
            if get_column(self.ordinaryGrid, col)[row] == 0:
                col_free_cells.append(row)

        return col_free_cells

    def find_valid_cells_in_row(self, row, ord_instances):
        valid_cols = self.find_valid_cols_for_num(ord_instances)
        row_valid_cells = []

        row_free_cells = self.find_free_cells_in_row(row)

        for row_cell in row_free_cells:
            if row_cell in valid_cols:
                row_valid_cells.append(row_cell)

        return row_valid_cells

    def find_valid_cells_in_col(self, col, ord_instances):
        valid_rows = self.find_valid_rows_for_num(ord_instances)
        col_valid_cells = []

        col_free_cells = self.find_free_cells_in_col(col)  # Correct
        for col_cell in col_free_cells:
            if col_cell in valid_rows:
                col_valid_cells.append(col_cell)

        return col_valid_cells

    def add_to_row(self, number, row, row_valid_cells):
        if len(row_valid_cells) == 1:
            self.add_num_in_ord(number, row, row_valid_cells[0])

    def add_to_col(self, number, col, col_valid_cells):
        if len(col_valid_cells) == 1:
            self.add_num_in_ord(number, col_valid_cells[0], col)

    ##########################################################################################
    # ------------------------------Solving by Cell-------------------------
    ##########################################################################################
    def solve_by_cell_in_row(self, row):
        empty_cells = []
        for cell in range(0, 9):
            if self.ordinaryGrid[row][cell] == 0:
                empty_cells.append(cell)

        missing_numbers = self.get_missing_numbers_row(row)
        k = 0

        missings = []
        if len(missing_numbers) == 3:
            for cell in empty_cells:
                if k == 1:
                    break
                for num in missing_numbers:
                    for i in range(3):
                        missings.append(missing_numbers[i])
                    missings.remove(num)
                    instances1 = self.find_ord_instances_for_num(missings[0])
                    instances2 = self.find_ord_instances_for_num(missings[1])

                    if cell in get_column(instances1, 1) and cell in get_column(instances2, 1):
                        self.add_num_in_ord(num, row, cell)
                        k = 1
                        break

    def solve_by_cell_in_col(self, col):
        empty_cells = []
        for cell in range(0, 9):
            if self.ordinaryGrid[cell][col] == 0:
                empty_cells.append(cell)
        missing_numbers = self.get_missing_numbers_col(col)
        k = 0
        missings = []
        if len(missing_numbers) == 3:
            for cell in empty_cells:
                if k == 1:
                    break
                for num in missing_numbers:
                    for i in range(3):
                        missings.append(missing_numbers[i])
                    missings.remove(num)
                    instances1 = self.find_ord_instances_for_num(missings[0])
                    instances2 = self.find_ord_instances_for_num(missings[1])
                    if cell in get_column(instances1, 0) and cell in get_column(instances2, 0):
                        self.add_num_in_ord(num, cell, col)
                        k = 1
                        break

    def get_missing_numbers_col(self, col):
        missing_numbers = []
        for num in range(1, 10):
            if num not in get_column(self.ordinaryGrid, col):
                missing_numbers.append(num)
        return missing_numbers

    def get_missing_numbers_row(self, row):
        missing_numbers = []
        for num in range(1, 10):
            if num not in self.ordinaryGrid[row]:
                missing_numbers.append(num)
        return missing_numbers


