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

    y = int(my_model.predict_classes(img)[0])

    return y


def digitize_captured(sudo, my_model):
    """
    Arguments:
    sudo -- A squared and unsolved sudoku image
    my_model -- A classification CNN model trained on 10 classes
                classes: blank images, 1, 2, .., 9

    Returns:
    grid -- a (9 x 9) numpy array of all predicted digits
    """

    delta_w, delta_h = int(sudo.shape[0] / 9), int(sudo.shape[1] / 9)
    resizes = []
    dd=5
    for h in range(9):
        for w in range(9):
#             crop = sudo[w * delta_w:(w * delta_w + delta_w),
#                    h * delta_h:(h * delta_h + delta_h)]
            crop = sudo[w * delta_w+dd:(w*delta_w + delta_w-dd), h*delta_h+dd:(h * delta_h + delta_h-dd)]
            resize = cv2.resize(crop, (28, 28), interpolation=cv2.INTER_AREA).reshape(28, 28, 1)
            resizes.append(resize)

    resizes = np.array(resizes)
    y = my_model.predict_on_batch(resizes)
    digits = np.argmax(y, axis=1)
    grid = digits.reshape((9, 9), order='F')
    return grid.tolist(), y






'''
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
    # y = my_model.predict(img)
    # digit = int(np.argmax(y, axis=1)[0])

    y = int(my_model.predict_classes(img)[0])

    # TODO - disregard low probability digits
    return y


def digitize_captured(sudo, my_model):
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
    # i = 0
    for h in range(9):
        for w in range(9):
            crop = sudo[w*delta_w : (w*delta_w+delta_w), h*delta_h : (h*delta_h+delta_h)]
            # cv2.imwrite('Testing/img_'+str(i)+'.jpg', crop)  #
            # i += 1
            resize = cv2.resize(crop, (28, 28), interpolation=cv2.INTER_AREA).reshape(1, 28, 28, 1)
            grid[w][h] = predict_digit(resize, my_model)
    return grid.tolist()










'''



