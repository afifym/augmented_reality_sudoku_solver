"""
Author: Mohamed Afify
Date: June 2020

Before reading:
    1. For the sake of making things easy to understand:
        sudoku -- the unsolved image of the sudoku
        sudoku_sol -- the solved image of the sudoku
        grid -- a matrix representing the digitized and unsolved sudoku
        grid_sol -- a matrix representing the digitized and solved sudoku
    2. So this is generally how it works:
        image >> sudoku >> grid >> sudoku_sol >> grid_sol >> augmented_image

"""


import capture
import digitize
import solve
import numpy as np
import cv2
from tensorflow import keras


def augment_to_image(num_img, points, image):
    src = capture.dst
    matrix = cv2.getPerspectiveTransform(src, points)
    aug_nums = cv2.warpPerspective(num_img, matrix, (640, 360))
    aug_img = cv2.addWeighted(image, 1, aug_nums, 1, 1)
    return aug_img


def get_sudo_ku(grid_sol, sudoku, zero_indices):
    delta_w, delta_h = int(sudoku.shape[0] / 9), int(sudoku.shape[1] / 9)
    sudoku_sol = np.zeros((sudoku.shape[0], sudoku.shape[1], 3), np.uint8)
    for h in range(9):
        for w in range(9):
            if [h, w] in zero_indices:
                if int(grid_sol[h][w]) != 0:
                    cv2.putText(sudoku_sol, str(grid_sol[h][w]),
                                (w*delta_w + to_center-2, h*delta_h + to_center+10),
                                font, font_size, (0, 255, 0))
    return sudoku_sol


params = {'sudokuBlur':7,
          'sudokuThresh':11,
          'solveLevel':30,
          'fontSize':0.9
}

my_model = keras.models.load_model("Models/digitizer_noise_99acc.h5")
font = cv2.FONT_HERSHEY_SIMPLEX
font_size = params['fontSize']
to_center = 18
augmented = np.zeros((450, 450))
cap = cv2.VideoCapture(0)
i = 0

while True:
    matrix, sudoku, pts, img, detected = capture.capture_sudo(cap)  # whole image >> squared unsolved sudoku image

    ui_string, ui_color = "Put the sudoku in the center", (0, 0, 255)
    sudoku_thresh = sudoku
    if detected:
        sudoku_blur = cv2.GaussianBlur(sudoku, (params['sudokuBlur'], params['sudokuBlur']), 0)
        sudoku_thresh = cv2.adaptiveThreshold(sudoku_blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, params['sudokuThresh'], 4)

        grid = digitize.digitize_captured(sudoku_thresh / 255, my_model)
        grid_solved, zero_indices = solve.solve_grid(grid, params['solveLevel'])  # a grid of digits >> a solved sudoku grid
        sudoku_sol = get_sudo_ku(grid_solved, sudoku, zero_indices)  # a solved sudoku grid >> squared solved sudoku image
        augmented = augment_to_image(sudoku_sol, pts, img)  # squared solved sudoku image >> augmented image
        ui_string, ui_color = "Sudoku Detected!", (0, 255, 0)

    cv2.putText(img, ui_string, (10, 350), font, 0.6, ui_color)
    cv2.line(img, (320, 0), (320, 360), ui_color, 1)
    cv2.line(img, (0, 180), (640, 180), ui_color, 1)

    cv2.imshow('Normal', img)
    cv2.imshow('Sudoku', sudoku_thresh)
    cv2.imshow('Result', augmented)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

