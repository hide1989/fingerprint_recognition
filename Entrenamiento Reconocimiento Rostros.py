import cv2
import os

import numpy as np

carpeta_entrada="imagenes/entrenamiento/rostros"

lista_rostros=os.listdir(carpeta_entrada)

datos_rostros=[]
etiquetas=[]
total_etiquetas=0

for rostro in lista_rostros:
    print("Procesando imagen: ", rostro)
    img=cv2.imread(os.path.join(carpeta_entrada, rostro), 0)

    if img is not None:
        datos_rostros.append(img)
        etiquetas.append(total_etiquetas)
        total_etiquetas+=1

reconocedor_rostro=cv2.face.LBPHFaceRecognizer_create()

reconocedor_rostro.train(datos_rostros, np.array(etiquetas))
print("Entrenando...")

reconocedor_rostro.write("reconocedor_rostro.xml")

print("Entrenamiento finalizado")


