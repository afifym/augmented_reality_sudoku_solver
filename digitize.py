import cv2
import numpy as np


def digitize_captured(sudo, my_model):
    """
    Arguments:
    sudo -- A squared and unsolved sudoku image
    my_model -- A classification CNN model trained on 10 classes
                classes: blank images, 1, 2, .., 9

    Returns:
    grid -- a (9 x 9) matrix of all predicted digits
    """

    delta_w, delta_h = int(sudo.shape[0] / 9), int(sudo.shape[1] / 9)   # Ratios to divide the image by
    resizes = []                  # an empty list where we will append the cropped cells
    dd = 5
    for h in range(9):
        for w in range(9):
            crop = sudo[w * delta_w+dd:(w*delta_w + delta_w-dd),
                        h*delta_h+dd:(h * delta_h + delta_h-dd)]        # extract a cell to predict its class
            resize = cv2.resize(crop, (28, 28), interpolation=cv2.INTER_AREA).reshape(28, 28, 1)
            resizes.append(resize)

    resizes = np.array(resizes)
    y = my_model.predict_on_batch(resizes)     # Perform batch prediction for all cells (faster than predicting one cell at a time)
    digits = np.argmax(y, axis=1)
    grid = digits.reshape((9, 9), order='F')
    return grid.tolist()



