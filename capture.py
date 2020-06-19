import numpy as np
import cv2
from numpy.linalg import norm

sudo_width, sudo_height = 450, 450
default_sudo = np.zeros((sudo_width, sudo_height, 3))
dst = np.array([[0, 0], [sudo_width - 1, 0], 
                [sudo_width - 1, sudo_height - 1], [0, sudo_height - 1]], np.float32)


def two_vector_mag(vector_1, vector_2):
    vector_1 = np.array(vector_1)
    vector_2 = np.array(vector_2)
    return np.abs(np.linalg.norm(vector_1 - vector_2))


def capture_sudo(cap):
    """
    Arguments:
    cap -- the camera object to be used for capturing images

    Returns:
    sudo -- A squared image of the extracted sudoku grid
    pts -- the 4 points describing the sudoku in img
    img -- the captured image (before any processing)
    """

    # 1. Capturing an image and finding all edges in that image
    ret, img = cap.read()         # Reading an image from camera
    img = img[60:-60, :]          # Crop the image to eliminate black borders (may depend on your camera)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (9, 9), 0)
    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                   cv2.THRESH_BINARY_INV, 11, 4)

    # 2. Finding the contour with the larges area (should constitute the sudoku)
    # then converting it to a four-point polygon
    _, contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnt_max = max(contours, key=cv2.contourArea)
    approx = cv2.approxPolyDP(cnt_max, 0.009 * cv2.arcLength(cnt_max, True), True)

    # 3. Extracting the 4 poly points and making sure they resemble a sudoku
    try:
        poly_points = [[approx[0][0][0], approx[0][0][1]],
                      [approx[1][0][0], approx[1][0][1]],
                      [approx[2][0][0], approx[2][0][1]],
                      [approx[3][0][0], approx[3][0][1]]]
        
        mag_diff1 = two_vector_mag(poly_points[0], poly_points[1]) - \
                    two_vector_mag(poly_points[2], poly_points[3])
        mag_diff2 = two_vector_mag(poly_points[0], poly_points[2]) - \
                    two_vector_mag(poly_points[1], poly_points[3])

        # Making sure the poly points are in the right order
        for point in poly_points:
            if point[0] < 320 and point[1] < 180:
                p0 = point
            elif point[0] > 320 and point[1] < 180:
                p1 = point
            elif point[0] < 320 and point[1] > 180:
                p2 = point
            elif point[0] > 320 and point[1] > 180:
                p3 = point
                
        cv2.circle(img, tuple(p0), 5, (0, 0, 255), -1)
        cv2.circle(img, tuple(p1), 5, (255, 0, 0), -1)
        cv2.circle(img, tuple(p2), 5, (255, 0, 0), -1)
        cv2.circle(img, tuple(p3), 5, (0, 0, 255), -1)

        # Checking whether the 4 points can resemble a sudoku
        limit = 20
        if mag_diff1 > limit or mag_diff2 > limit:
            return [], default_sudo, [], img, False
        
        pts = np.array([p0, p1, p3, p2], np.float32)
        
        try:
            # 4. Getting the rotation matrix then warp the poly region to extract a squared sudoku
            matrix = cv2.getPerspectiveTransform(pts, dst)
            sudo = cv2.warpPerspective(img, matrix, (sudo_width, sudo_height))
            sudo = cv2.cvtColor(sudo, cv2.COLOR_BGR2GRAY)

            return matrix, sudo, pts, img, True
        except:
            pass
    except:
        pass

    return [], default_sudo, [], img, False


