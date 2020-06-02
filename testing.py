
import digitize_sudo_inference as digitize
import cv2
from tensorflow import keras
import show_soduku_opencv as sh

# TODO - passed: 1, 8, 6

model_name = "digitizer_98_blank.h5"
sudo_name = "sudo_2.jpg"

my_model = keras.models.load_model('Models/' + model_name)
sudo = cv2.imread('Sudos/'+sudo_name, 0)
cv2.imwrite('sud.jpg', sudo)
grid = digitize.digitize_sudo(sudo, my_model)
zero_indices = sh.get_zeros(grid)
sh.draw_sudoku(grid, zero_indices)


