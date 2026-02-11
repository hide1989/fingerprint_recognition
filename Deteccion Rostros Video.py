import cv2

# Crear la cascada de haar 
ch = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

cv = cv2.VideoCapture(0)

while True:
    #captura una diapositiva
    ret, imgCapturada=cv.read()

    imgGris = cv2.cvtColor(imgCapturada, cv2.COLOR_BGR2GRAY)

    rostros = ch.detectMultiScale(
        imgGris,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags= cv2.CASCADE_SCALE_IMAGE
    )

    for (x, y, w, h) in rostros:
        cv2.rectangle(imgCapturada, (x, y), (x+w, y+h), (0,225, 0), 2)

    cv2.imshow("Video", imgCapturada)

    #tecla de salida
    if cv2.waitKey(1) & 0xFF == ord('s'):
        break

cv.release()






