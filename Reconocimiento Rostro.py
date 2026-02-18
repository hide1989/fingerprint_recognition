import cv2

reconocedor_rostro=cv2.face.LBPHFaceRecognizer_create()
reconocedor_rostro.read("reconocedor_rostro.xml")

img = cv2.imread('imagenes/Fray y Yeny.png')
imgGris = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

ch = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

rostros = ch.detectMultiScale(
    imgGris,
    scaleFactor=1.1,
    minNeighbors=5
    )

for (x, y, w, h) in rostros:
    rostro_recortado=imgGris[y:y+h, x:x+w]
    rostro_redimensionado=cv2.resize(rostro_recortado, (150, 150))

    resultado = reconocedor_rostro.predict(rostro_redimensionado)

    # resultado[0] es el ID
    # #resultado[1] es la distancia/confianza (a menor valor, más parecido)

    print(resultado[1])

    if resultado[1] < 75:
        cv2.rectangle(img, (x, y), (x+w, y+h), (0,225, 0), 2)
    else:
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 2)

cv2.imshow("Resultado Reconocimiento Facial", img)
cv2.waitKey(0)












