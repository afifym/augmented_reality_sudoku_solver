"""
Input: A squared and unsolved sudoku image
Output: A grid of digits (python array)
Method: Using a trained CNN model on 10 classes (blank white image, 1 through 9)

Process:
    1. Ensure having a suitable image input
    2. Train a CNN model (a 1 hidden layer model is a good start)
    3.
"""



import cv2
import numpy as np


def predict_digit(img, my_model):
    """
    Arguments:
    img -- An image of a cell (corresponds to 1 digit)
    my_model -- A classification CNN model trained on 10 classes
                classes: blank images, 1, 2, .., 9

    Returns:
    digit -- a single digit predicted in the current cell
    """

    y = my_model.predict(img/255)
    digit = int(np.argmax(y, axis=1)[0])
    return digit


def digitize_sudo(sudo, my_model):
    """
    Arguments:
    sudo -- A squared and unsolved sudoku image
    my_model -- A classification CNN model trained on 10 classes
                classes: blank images, 1, 2, .., 9

    Returns:
    grid -- a (9 x 9) numpy array of all predicted digits
    """

    grid = np.zeros((9, 9))
    delta_w, delta_h = int(sudo.shape[0]/9), int(sudo.shape[1]/9)
    i = 0
    for h in range(9):
        for w in range(9):
            crop = sudo[w*delta_w : (w*delta_w+delta_w), h*delta_h : (h*delta_h+delta_h)]
            cv2.imwrite('Testing/img_'+str(i)+'.jpg', crop)
            i += 1
            resize = cv2.resize(crop, (28, 28), interpolation=cv2.INTER_AREA).reshape(1, 28, 28, 1)
            grid[w][h] = predict_digit(resize, my_model)
    return grid










