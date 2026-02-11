import cv2

#crear la cascada de Haar
ch = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

#cargar la imagen
imgOriginal = cv2.imread("imagenes/Fray y Yeny.png")
#convertir a grises
imgGrises = cv2.cvtColor(imgOriginal, cv2.COLOR_BGR2GRAY)

#deterctar los rostros
rostros = ch.detectMultiScale(
    imgGrises,
    scaleFactor=1.1,
    minNeighbors=5,
    minSize=(30, 30),
    flags= cv2.CASCADE_SCALE_IMAGE
    )

#seleccionar rostros
for (x, y, w, h) in rostros:
    cv2.rectangle(imgOriginal, (x, y), (x+w, y+h), (0,225, 0), 2)

#mostrar resultado
cv2.imshow("Rostros encontrados", imgOriginal)
cv2.waitKey(0)
