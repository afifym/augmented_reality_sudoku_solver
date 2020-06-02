import capture_sudo
import digitize_sudo_inference as digitize
import solve
import numpy as np
import cv2
from tensorflow import keras


my_model = keras.models.load_model("Models/digitizer_95_mnist.h5")
cap = cv2.VideoCapture(0)
font = cv2.FONT_HERSHEY_SIMPLEX
font_size = 0.6
blue = (225, 144, 62)
num_color = blue
to_center = 18


def augment_to_image(sudo_ku, pts, img):
    dst = capture_sudo.dst
    matrix = cv2.getPerspectiveTransform(dst, pts)
    aug_sudo_ku = cv2.warpPerspective(sudo_ku, matrix, (capture_sudo.width, capture_sudo.height))

    return aug_img


def get_sudo_ku(sudoku, sudo_ku, zero_indices):
    delta_w, delta_h = int(sudo_ku.shape[0] / 9), int(sudo_ku.shape[1] / 9)
    for h in range(9):
        for w in range(9):
            if [h, w] in zero_indices:
                cv2.putText(sudo_ku, str(sudoku[h, w]), ((w*delta_w) + to_center, (h*delta_h) + to_center),
                            font, font_size, num_color)

    return sudo_ku


while True:
    sudo, pts, img = capture_sudo.capture(cap)        # whole image >> squared unsolved sudoku image
    grid = digitize.digitize_sudo(sudo, my_model)      # squared unsolved sudoku image >> a grid of digits
    sudoku, zero_indices = solve.solve_grid(grid.tolist())      # a grid of digits >> a solved sudoku grid
    # TODO: add leveling
    sudo_ku = get_sudo_ku(sudoku, sudo, zero_indices)        # a solved sudoku grid >> squared solved sudoku image
    augmented = augment_to_image(sudo_ku, pts, img)     # squared solved sudoku image >> augmented image

    cv2.imshow('Result', augmented)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

