import numpy as np
import cv2

width, height = 640, 480
sudo_width, sudo_height = 450, 450
sudo = np.zeros((sudo_width, sudo_height, 3))
dst = np.array([[0, 0], [sudo_width - 1, 0], [sudo_width - 1, sudo_height - 1], [0, sudo_height - 1]], np.float32)


def capture(cap):
    """
    Arguments:
    cap -- the camera object to be used for capturing images

    Returns:
    sudo -- A squared image of the extracted sudoku grid
    pts -- the 4 points describing the sudoku in img
    img -- the captured image (before any processing)
    """

    ret, img = cap.read()
    img = img[60:-60, :]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (15, 15), 0)
    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                   cv2.THRESH_BINARY_INV, 11, 4)

    _, cnts, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnt_max = max(cnts, key=cv2.contourArea)
    approx = cv2.approxPolyDP(cnt_max, 0.009 * cv2.arcLength(cnt_max, True), True)

    try:
        cv2.circle(img, tuple([approx[0][0][0], approx[0][0][1]]), 10, (255, 0, 0), -1)
        cv2.circle(img, tuple([approx[1][0][0], approx[1][0][1]]), 10, (0, 0, 255), -1)
        cv2.circle(img, tuple([approx[2][0][0], approx[2][0][1]]), 10, (255, 0, 0), -1)
        cv2.circle(img, tuple([approx[3][0][0], approx[3][0][1]]), 10, (0, 0, 255), -1)
        pts = np.array([approx.tolist()[0][0], approx.tolist()[1][0], approx.tolist()[2][0], approx.tolist()[3][0]],
                       np.float32)
        try:
            matrix = cv2.getPerspectiveTransform(pts, dst)
            sudo = cv2.warpPerspective(img, matrix, (sudo_width, sudo_height))
            return sudo, pts, img
        except:
            pass
    except:
        pass

    return np.zeros((sudo_width, sudo_height, 3)), [], img


