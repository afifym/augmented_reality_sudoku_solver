import testing
import cv2
import numpy as np

while True:
    cv2.imshow('window', testing.capture())

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

testing.cap.release()
cv2.destroyAllWindows()

