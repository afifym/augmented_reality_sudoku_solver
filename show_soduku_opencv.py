"""
Author: Mohamed Afify
Date: May 2020

- This file displays a sodoku and shows the added numbers
- Can be used when testing the solving algorithm on grids (like the ones in grids.py)
    before using a camera.
- Uses opencv
"""

import cv2


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


def draw_sudoku(to_draw, zeros_indices, save=False):
    """
    Arguments:
    to_draw -- A (9 x 9) array (sudoku grid to be shown)
    zeros_indices -- A list of coordinates of originally empty sudoku cells (before solving)
    """

    window_width, window_height = 450, 450
    black = (0, 0, 0)
    white = (200, 200, 200)
    gray = (50, 50, 50)
    blue = (225, 144, 62)
    rect_color = blue
    num_color = black

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_size = 0.65

    sudoku = cv2.imread('sudoku-01.bmp')

    margin = 50
    i = 0
    to_center = 18

    while True:
        if i == 0:
            for row in to_draw:
                j = 0
                for num in row:
                    c = int(i/50)
                    r = int(j/50)
                    if [c, r] in zeros_indices:
                        cv2.rectangle(sudoku, (r*margin, c*margin), (r*margin+margin, c*margin+margin), rect_color, -1)
                    if num == 0:
                        cv2.putText(sudoku, ' ', (j + to_center, i + to_center*2 - 4), font, font_size, num_color)
                    else:
                        cv2.putText(sudoku, str(int(num)), (j + to_center, i + to_center*2 - 4), font, font_size, num_color)
                    j += margin
                i += margin
            if save:
                cv2.imwrite('result.jpg', sudoku)

        cv2.imshow('Result', sudoku)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()