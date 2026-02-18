import cv2

reconocedor_rostro=cv2.face.LBPHFaceRecognizer_create()
reconocedor_rostro.read("reconocedor_rostro.xml")

img = cv2.imread('imagenes/Imagen Prueba.png')
imgGris = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

ch = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

rostros = ch.detectMultiScale(
    imgGris = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY),
    scaleFactor=1.1,
    minNeighbors=5,
    minSize=(30, 30),
    flags= cv2.CASCADE_SCALE_IMAGE
    )

for (x, y, w, h) in rostros:
    rostro_recortado=img[y:y+h, x:x+w]
    rostro_redimensionado=cv2.resize(rostro_recortado, (150, 150))




