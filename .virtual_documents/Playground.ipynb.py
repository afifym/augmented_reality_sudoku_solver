import cv2
import numpy as np

# cap = cv2.VideoCapture(cv2.CAP_DSHOW)
cap = cv2.VideoCapture(0)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT-100, 480)
mm = 1

sudoku = np.zeros((400, 400), np.uint8)

while(True):
    ret, frame = cap.read()
    img = frame[60:-60, :]
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    blur = cv2.GaussianBlur(gray, (15, 15), 0)
    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                   cv2.THRESH_BINARY_INV, 11, 4)
    
#     kernel = np.ones((15,15), np.float32)/25
#     blur = cv2.filter2D(gray, -1, kernel)
#     blur = cv2.bilateralFilter(gray, 9, 75, 75)
    canny = cv2.Canny(gray, 50, 100, apertureSize=3)
    
    using = thresh

    using_bgr = cv2.cvtColor(using, cv2.COLOR_GRAY2BGR)
    cv2.imshow('Before', using_bgr)
    lines = cv2.HoughLinesP(using, 1, np.pi/180, 1, minLineLength=2, maxLineGap=5)

    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(using_bgr, (x1, y1), (x2, y2), (0, 255, 0), 1)
        
    _, cnts, _ = cv2.findContours(cv2.cvtColor(using_bgr, cv2.COLOR_BGR2GRAY),
                                  cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnt_max = max(cnts, key=cv2.contourArea)
    
    epsilon = 0.001*cv2.arcLength(cnt_max, True)
    approx = cv2.approxPolyDP(cnt_max, epsilon, True)
    
#     cv2.polylines(using_bgr, approx, False, (255, 255, 255))
    
#     x, y, w, h = cv2.boundingRect(approx)
#     cv2.rectangle(using_bgr, (x, y), (x+w, y+h), (0, 0, 255), 2)
    
#     mask_value = 255
#     fill_color = 0
#     stencil = np.zeros(img.shape[:-1]).astype(np.uint8)
#     cv2.fillConvexPoly(stencil, approx, mask_value)
#     sel = stencil get_ipython().getoutput("= mask_value # select everything that is not mask_value")
#     img[sel] = fill_color            # and fill it with fill_color
    

    mask = np.zeros((img.shape[0], img.shape[1]))
    cv2.fillConvexPoly(mask, approx, 1)
    mask = mask.astype(np.bool)
    out = np.zeros_like(img)
    out[mask] = img[mask]
    cv2.imshow('After', out)
    
#     cv2.drawContours(using_bgr, c, contourIdx=-1, color=(255, 255, 255),
#                      thickness=-1)

    n = approx.ravel()
    i = 0
    x = []
    y = []
    for j in n:
        if(i % 2 == 0):
            x.append(n[i])
            y.append(n[i + 1])
        i = i + 1

#     print(len(x))
#     print(len(y))
    print(np.shape(np.array(cnts)))
#     print('--done--')
#     pts1 = np.float32([[56,65],[368,52],[28,387],[389,390]])
#     pts2 = np.float32([[0,0],[300,0],[0,300],[300,300]])

#     matrix = cv2.getPerspectiveTransform(pts1, pts2)
#     result = cv2.warpPerspective(frame, matrix, (500, 600))
#     cv2.imshow('Transform', result)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


import numpy as np 
import cv2 
from numpy.linalg import norm

dist = np.linalg.norm(a-b)


[np.linalg.norm(np.array(p0)-np.array(p1)), np.linalg.norm(np.array(p2)-np.array(p3))]


[np.linalg.norm(np.array(p0)-np.array(p2)), np.linalg.norm(np.array(p1)-np.array(p3))]


num_img.shape


num_img = np.ones((360, 640))
bgr = cv2.cvtColor(cv2.flip(num_img, 1), cv2.COLOR_GRAY2BGR)


import numpy as np 
import cv2 
from numpy.linalg import norm


def two_vector_mag(vector_1, vector_2):
    return np.abs(np.linalg.norm(vector_1 - vector_2))


def two_vector_ang(vector_1, vector_2):
    unit_v1 = vector_1 / np.linalg.norm(vector_1)
    unit_v2 = vector_2 / np.linalg.norm(vector_2)
    return np.arccos(np.dot(unit_v1, unit_v2))


cap = cv2.VideoCapture(0)
font = cv2.FONT_HERSHEY_COMPLEX 

width, height = 640, 360
sudo_width, sudo_height = 450, 450
sudo = np.zeros((sudo_width, sudo_height, 3))
dst = np.array([[0, 0], [sudo_width - 1, 0], [sudo_width - 1, sudo_height - 1], [0, sudo_height - 1]], np.float32)
M = cv2.getRotationMatrix2D((sudo_width/2, sudo_height/2), 90, 1.0)

while(True):
    ret, img = cap.read()
    img = img[60:-60, :]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # ----------------
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 11, 4)
    # ----------------
    
    _, cnts, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnt_max = max(cnts, key=cv2.contourArea)

    approx = cv2.approxPolyDP(cnt_max, 0.009 * cv2.arcLength(cnt_max, True), True)

    try:
        p0 = np.array([approx[0][0][0], approx[0][0][1]])
        p1 = np.array([approx[1][0][0], approx[1][0][1]])
        p2 = np.array([approx[2][0][0], approx[2][0][1]])
        p3 = np.array([approx[3][0][0], approx[3][0][1]])            

        cv2.circle(img, tuple(p0), 10, (255, 0, 0), -1)
        cv2.circle(img, tuple(p1), 10, (0, 0, 255), -1)
        cv2.circle(img, tuple(p2), 10, (0, 255, 0), -1)
        cv2.circle(img, tuple(p3), 10, (0, 255, 255), -1)
                
        mag_diff1 = two_vector_mag(p0, p1) - two_vector_mag(p2, p3)
        mag_diff2 = two_vector_mag(p0, p2) - two_vector_mag(p1, p3)
        ang_diff1 = two_vector_ang(p0, p1) - two_vector_ang(p2, p3)
        ang_diff2 = two_vector_ang(p0, p2) - two_vector_ang(p1, p3)
        
        
        if mag_diff1 > 5 or mag_diff2 > 5 or ang_diff1 < 0.5 or ang_diff2 < 0.5:
            pass
        else:
            print('Breaking')
            
        pts = np.array([approx.tolist()[0][0], approx.tolist()[1][0],
                        approx.tolist()[2][0], approx.tolist()[3][0]], np.float32)
        
        try:
            matrix = cv2.getPerspectiveTransform(pts, dst)
            sudo = cv2.warpPerspective(img, matrix, (sudo_width, sudo_height))
            sudo = cv2.flip(sudo, 1)
        
            if matrix[0, 2] < 0:
                sudo = cv2.warpAffine(sudo, M, (sudo_height, sudo_width))

#             if matrix[2, 0]<0 and matrix[0, 2]<0:

        except:
                pass
    except:
        pass
    
    cv2.imshow('Sudoku', sudo)
    cv2.imshow('Original', img)
    cv2.drawContours(thresh, [cnt_max], 0, (255, 255, 255), -1)
    cv2.imshow('thresh', thresh)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


# Normal
matrix


matrix


import glob
import cv2
import numpy as np

files = glob.glob("Sudokus/sudos/*.jpg")
# print(files)
i = 0

for img in files:
    sudo = cv2.imread(img)
    print(img)
    delta_w, delta_h = int(sudo.shape[0] / 9), int(sudo.shape[1] / 9)
    for h in range(9):
        for w in range(9):
            crop = sudo[w*delta_w : (w*delta_w+delta_w), h*delta_h : (h*delta_h+delta_h)]
            cv2.imwrite('Sudokus/generated/img_'+str(i)+'.jpg', crop)
            i = i+1


import random
import os
from shutil import copyfile


def split_data(SOURCE, TRAINING, VALIDATION, SPLIT_SIZE):
    files = []
    for filename in os.listdir(SOURCE):
        file = SOURCE + filename
        if os.path.getsize(file) > 0:
            files.append(filename)
        else:
            print(filename + " is zero length, so ignoring.")

    training_length = int(len(files) * SPLIT_SIZE)
    validation_length = int(len(files) - training_length)
    shuffled_set = random.sample(files, len(files))
    training_set = shuffled_set[0:training_length]
    validation_set = shuffled_set[-validation_length:]

    for filename in training_set:
        this_file = SOURCE + filename
        destination = TRAINING + filename
        copyfile(this_file, destination)

    for filename in validation_set:
        this_file = SOURCE + filename
        destination = VALIDATION + filename
        copyfile(this_file, destination)


source_dir = 'dataset/source/'
training_dir = 'dataset/training/'
validation_dir = 'dataset/validation/'



for i in range(1, 10):
    src_dir = source_dir + 'num_' + str(i) + '/'
    train_dir = training_dir + 'num_' + str(i) + '/'
    valid_dir = validation_dir + 'num_' + str(i) + '/'
    split_data(src_dir, train_dir, valid_dir, 0.9)


src_dir = source_dir + 'blank/'
train_dir = training_dir + 'blank/'
valid_dir = validation_dir + 'blank/'
split_data(src_dir, train_dir, valid_dir, 0.9)


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator

TRAINING_DIR = "dataset/source/"

train_datagen = ImageDataGenerator(validation_split=0.2,
                                   rescale=1./255)


train_batches = train_datagen.flow_from_directory(TRAINING_DIR,
                                                  target_size=(28, 28),
                                                  color_mode='grayscale',
                                                  class_mode='categorical',
                                                  batch_size=16,
                                                  subset='training')

valid_batches = train_datagen.flow_from_directory(TRAINING_DIR,
                                                  target_size=(28, 28),
                                                  color_mode='grayscale',
                                                  class_mode='categorical',
                                                  batch_size=16,
                                                  subset='validation')


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator


TRAINING_DIR = "dataset/dataset/training/"
VALIDATION_DIR = "dataset/dataset/validation/"

train_datagen = ImageDataGenerator(rescale=1./255)
train_batches = train_datagen.flow_from_directory(TRAINING_DIR,
                                                  target_size=(28, 28),
                                                  color_mode='grayscale',
                                                  class_mode='categorical',
                                                  batch_size=32)


valid_datagen = ImageDataGenerator(rescale=1./255)
valid_batches = valid_datagen.flow_from_directory(VALIDATION_DIR,
                                                  target_size=(28, 28),
                                                  color_mode='grayscale',
                                                  class_mode='categorical',
                                                  batch_size=32)


# per = train_batches.index_array
# classes = train_batches.classes[per]
# print(classes)
# .class_indices
# .filenames
train_batches.class_indices


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D

num_classes = 10

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['acc'])


class Cally(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):  # a special method in keras.callbacks
        if(logs.get('acc') > 0.95):          # termination condition
            print('\nReached 95% Accuracy, terminatingget_ipython().getoutput("\n')")
            self.model.stop_training = True    # terminates the training
CB = Cally()


history = model.fit_generator(generator=train_batches,
                              steps_per_epoch=train_batches.n//train_batches.batch_size,
                              epochs=100,
                              validation_data=valid_batches,
                              callbacks=[CB])

# model.save("digits_finder_2.h5")



from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())


sess = tf.compat.v1.Session(config=tf.ConfigProto(log_device_placement=True))


get_ipython().run_line_magic("matplotlib", " inline")
# import matplotlib.image  as mpimg
import matplotlib.pyplot as plt

acc=history.history['acc']
val_acc=history.history['val_acc']
loss=history.history['loss']
val_loss=history.history['val_loss']

epochs=range(len(acc))

plt.plot(epochs, acc, 'r', "Training Accuracy")
plt.plot(epochs, val_acc, 'b', "Validation Accuracy")
plt.title('Training and validation accuracy')
plt.figure()
plt.plot(epochs, loss, 'r', "Training Loss")
plt.plot(epochs, val_loss, 'b', "Validation Loss")
plt.figure()


my_model = keras.models.load_model("digitizer_95.h5")


img = cv2.imread('sudo_2.jpg', 0)/255
delta = int(img.shape[0]/9)
crop = img[5:delta, 5:delta]
y_pred = model.predict(preprocess(crop))
print(y_pred)
print(np.where(y_proba == np.amax(y_proba))[0][0])


import numpy as np
arr = []

if arr:
    print('hi')
else:
    print('ww')
