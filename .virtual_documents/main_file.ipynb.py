import tensorflow as tf
from tensorflow import keras
import cv2
import numpy as np
import timeit

model = keras.models.load_model("Models/digitizer_98_mnist.h5")
sudo = cv2.imread('Sudos/sudo_1.jpg', 0)/255
delta_w, delta_h = int(sudo.shape[0]/9), int(sudo.shape[1]/9)
resizes = []
for h in range(9):
    for w in range(9):
        crop = sudo[w*delta_w:(w*delta_w+delta_w),
                    h*delta_h:(h*delta_h+delta_h)]
        resize = cv2.resize(crop, (28, 28), interpolation=cv2.INTER_AREA).reshape(28, 28, 1)
        resizes.append(resize)


resizes = np.array(resizes)
print(resizes.shape)
start = timeit.default_timer()
y = model.predict_on_batch(resizes)
stop = timeit.default_timer()
print('Digitizing: ', stop - start)
digits = np.argmax(y, axis=1)
grid = digits.reshape((9, 9), order='F')


grid.tolist()


import capture
import digitize
import solve
import numpy as np
import cv2

import tensorflow as tf
from tensorflow import keras
import timeit


font = cv2.FONT_HERSHEY_SIMPLEX
font_size = 0.9
to_center = 18


def augment_to_image(num_img, points, image):
    src = capture.dst
    matrix = cv2.getPerspectiveTransform(src, points)
#     flip = cv2.flip(num_img, 1)
    aug_nums = cv2.warpPerspective(num_img, matrix, (capture.width, capture.height))
    aug_img = cv2.addWeighted(image, 1, aug_nums, 1, 1)
    return aug_img


def get_sudo_ku(sudoku, sudo_ku, zero_indices):
    delta_w, delta_h = int(sudo_ku.shape[0] / 9), int(sudo_ku.shape[1] / 9)
    num_img = np.zeros((sudo_ku.shape[0], sudo_ku.shape[1], 3), np.uint8)
    for h in range(9):
        for w in range(9):
            if [h, w] in zero_indices:
                if int(sudoku[h][w]) get_ipython().getoutput("= 0:")
                    cv2.putText(num_img, str(sudoku[h][w]),
                                (w*delta_w + to_center, h*delta_h + to_center+7),
                                font, font_size, (0, 255, 0))
    return num_img


my_model = keras.models.load_model("Models/digitizer_noise_99acc.h5")
augmented = np.zeros((450, 450))

cap = cv2.VideoCapture(0)
i = 0
while True:
    matrix, sudo, pts, img, detected = capture.capture_sudo(cap)        # whole image >> squared unsolved sudoku image
    
    ui_string, ui_color = "Put the sudoku in the center", (0, 0, 255)
    sudo_thresh = sudo
    if detected:
        sudo_blur = cv2.GaussianBlur(sudo, (11, 11), 0)
        sudo_thresh = cv2.adaptiveThreshold(sudo_blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 4)
        
        grid = digitize.digitize_captured(sudo_thresh/255, my_model)[0]      # squared unsolved sudoku image >> a grid of digits
        sudoku, zero_indices = solve.solve_grid(grid, 30)   # a grid of digits >> a solved sudoku grid
        sudo_ku = get_sudo_ku(sudoku, sudo, zero_indices)        # a solved sudoku grid >> squared solved sudoku image
        augmented = augment_to_image(sudo_ku, pts, img)     # squared solved sudoku image >> augmented image
        ui_string, ui_color = "Sudoku Detectedget_ipython().getoutput("", (0, 255, 0)")
        cv2.imwrite('test/t_'+str(i)+'.jpg', sudo_thresh)
        i+=1
    
    cv2.putText(img, ui_string, (10, 350), font, 0.6, ui_color)    
    cv2.line(img, (320, 0), (320, 360), ui_color, 1)
    cv2.line(img, (0, 180), (640, 180), ui_color, 1)
    
    cv2.imshow('Normal', img)
    cv2.imshow('Sudoku', sudo_thresh)
    cv2.imshow('Result', augmented)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


import os
import tensorflow as tf

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = ''

if tf.test.gpu_device_name():
    print('GPU found')
else:
    print("No GPU found")

# No GPU found


import digitize
import cv2
from tensorflow import keras
import show_soduku_opencv as sh
import solve
import timeit
import numpy as np

model_name = "digitizer_999acc.h5"
my_model = keras.models.load_model('Models/' + model_name)
sudo = cv2.imread('test/t_30.jpg', 0)
grid, y = digitize.digitize_captured(sudo, my_model)
acc = np.max(y, axis=1)
print('Average Accuracy: ', sum(acc)/81)
grid, zero_indices = solve.solve_grid(grid, 30)
sh.draw_sudoku(grid, zero_indices)


import digitize
import cv2
from tensorflow import keras
import show_soduku_opencv as sh
import solve
import timeit
import numpy as np
import glob

model_name = "digitizer_999acc.h5"
my_model = keras.models.load_model('Models/' + model_name)
i=0
files = glob.glob("test/*.jpg")
# sudo_name = "t_15.jpg"
for img in files:
    sudo = cv2.imread(img, 0)
    grid, y = digitize.digitize_captured(sudo, my_model)
    acc = np.max(y, axis=1)
    if sum(acc)/81 get_ipython().getoutput("= 1.0:")
        i+=1
        print('Average Accuracy: ', sum(acc)/81)
        print(img)
    
# grid, zero_indices = solve.solve_grid(grid, 30)
# sh.draw_sudoku(grid, zero_indices)


import cv2
import numpy as np
import glob
from tensorflow import keras
model_name = "digitizer_999acc.h5"
my_model = keras.models.load_model('Models/' + model_name)

files = glob.glob("test/*.jpg")
i=0
for img in files:
    sudo = cv2.imread(img, 0)
    delta_w, delta_h = int(sudo.shape[0] / 9), int(sudo.shape[1] / 9)
    dd = 5
    
    for h in range(9):
        for w in range(9):
            crop = sudo[w * delta_w+dd:(w*delta_w + delta_w-dd), h*delta_h+dd:(h * delta_h + delta_h-dd)]
            resize = cv2.resize(crop, (28, 28), interpolation=cv2.INTER_AREA).reshape(1, 28, 28, 1)
            y = int(my_model.predict_classes(resize)[0])
            cv2.imwrite('new/'+str(y)+'/c_'+str(i)+'.jpg', crop)
            i+=1
