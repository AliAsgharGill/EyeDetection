import cv2

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml') # pylint: disable=no-member
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml') # pylint: disable=no-member

img = cv2.imread('person.jpg') # pylint: disable=no-member
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # pylint: disable=no-member

faces = face_cascade.detectMultiScale(gray, 1.3, 5)

for (x, y, w, h) in faces:
    # cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2) # pylint: disable=no-member
    roi_gray = gray[y:y+h, x:x+w]
    roi_color = img[y:y+h, x:x+w]

    eyes = eye_cascade.detectMultiScale(roi_gray)
    for (ex, ey, ew, eh) in eyes:
        cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2) # pylint: disable=no-member

cv2.imshow('img', img) # pylint: disable=no-member
cv2.waitKey(0) # pylint: disable=no-member
cv2.destroyAllWindows() # pylint: disable=no-member
