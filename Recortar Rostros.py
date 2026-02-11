import cv2
import os

carpeta_entrada="imagenes/entrenamiento"
carpeta_salida="imagenes/entrenamiento/rostros"

ch = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

if not os.path.exists(carpeta_salida):
    print("Carpeta creada: ", carpeta_salida)
    os.makedirs(carpeta_salida)

total_recortes = 0
for nombre_imagen in os.listdir(carpeta_entrada):
    
    img = cv2.imread(os.path.join(carpeta_entrada, nombre_imagen))

    if img is None: continue

    print("Procesando imagen: ", nombre_imagen)

    imgGris = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    rostros = ch.detectMultiScale(
        imgGris,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags= cv2.CASCADE_SCALE_IMAGE
        )
    
    for (x, y, w, h) in rostros:
        #obtener el recorte con el rostro
        rostro_recortado=img[y:y+h, x:x+w]
        #redimensionar a un tamaños estandar
        rostro_redimensionado=cv2.resize(rostro_recortado, (150, 150))

        #guardar la imagen recortada
        total_recortes+=1
        cv2.imwrite(os.path.join(carpeta_salida, f"rostro_{total_recortes}.jpg"), rostro_redimensionado)




