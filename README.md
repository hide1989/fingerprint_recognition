# Reconocimiento de Huellas Dactilares con ORB

Sistema de reconocimiento de huellas dactilares basado en descriptores binarios **ORB** (Oriented FAST and Rotated BRIEF), binarización adaptativa gaussiana y la **prueba de ratio de Lowe** para filtrado de coincidencias.

---

## Requisitos

```bash
pip install opencv-contrib-python numpy
```

---

## Estructura del proyecto

```
fingerpirnt_detection/
├── fingerprint_recognition.py        # Script principal
├── fingerprint_images/               # Dataset: 80 imágenes de 10 sujetos (101–110)
│   ├── 101_1.tif
│   ├── 101_2.tif
│   └── ...
└── README.md
```

---

## Cómo ejecutar

Desde la raíz del proyecto:

```bash
python fingerpirnt_detection/fingerprint_recognition.py
```

El script selecciona aleatoriamente una imagen como consulta, la compara contra el resto del dataset y abre una ventana de visualización con el resultado.

---

## Pipeline de reconocimiento

```
Imagen .tif
    │
    ▼
1. Binarización adaptativa gaussiana   ← preprocesa cada imagen
    │
    ▼
2. Extracción de características ORB   ← detecta keypoints y descriptores binarios
    │
    ▼
3. Matching con BFMatcher (Hamming)    ← compara descriptores bit a bit
    │
    ▼
4. Prueba de ratio de Lowe (0.75)      ← descarta coincidencias ambiguas
    │
    ▼
5. Agregación de puntuaciones          ← suma matches por sujeto
    │
    ▼
6. Predicción                          ← sujeto con mayor puntuación total
```

### Binarización adaptativa

En lugar de un umbral global, cada píxel se compara contra la media ponderada gaussiana de su vecindario local (`ADAPTIVE_BLOCK × ADAPTIVE_BLOCK`). Esto preserva el detalle de crestas y bifurcaciones independientemente de la iluminación de la imagen.

### Prueba de ratio de Lowe

Para cada descriptor de la imagen consulta, se buscan los 2 vecinos más cercanos en el candidato (`m` y `n`). La coincidencia se acepta únicamente si:

```
distancia(m) < RATIO_TEST × distancia(n)
```

Un valor de `RATIO_TEST = 0.75` significa que el mejor vecino debe estar al menos un 25% más cerca que el segundo. Esto elimina coincidencias ambiguas y reduce los falsos positivos.

---

## Parámetros configurables

| Parámetro        | Valor por defecto | Descripción                                              |
|------------------|:-----------------:|----------------------------------------------------------|
| `ORB_NFEATURES`  | `500`             | Máximo de keypoints extraídos por imagen                 |
| `RATIO_TEST`     | `0.75`            | Umbral de la prueba de ratio (menor = más estricto)      |
| `ADAPTIVE_BLOCK` | `11`              | Tamaño del vecindario para la binarización (debe ser impar) |
| `ADAPTIVE_C`     | `2`               | Constante sustraída de la media ponderada                |

---

## Análisis de limitaciones

### Causas de puntajes bajos

1. **Rotación elevada** — ORB tolera rotaciones hasta ~30°. Por encima de 45°–60° la precisión cae significativamente.
2. **Brillo o contraste extremo** — Imágenes muy oscuras o muy brillantes en escala de grises dificultan la detección de minucias (puntos de interés).
3. **Ruido en la imagen** — Sudor, suciedad o presión irregular al capturar la huella puede introducir crestas o bifurcaciones inexistentes.
4. **Algoritmo de propósito general** — ORB no fue diseñado específicamente para huellas dactilares. Algoritmos especializados como los definidos en el estándar **ISO/IEC 19794** ofrecen mayor precisión.
5. **Ausencia de filtros de calidad previos** — Descartar imágenes con menos keypoints de los mínimos necesarios mejoraría la precisión al eliminar capturas de baja calidad.

---

## Escalabilidad

El sistema fue diseñado y validado con 80 imágenes (10 sujetos × 8 muestras). Para escalar a 1,000 sujetos (~8,000 imágenes) se deben considerar los siguientes aspectos:

### Almacenamiento
Las imágenes son objetos binarios, no datos estructurados. Servicios de object storage como **Amazon S3** (AWS) o **Azure Blob Storage** son la opción más adecuada para acceso frecuente a gran escala.

### Reducción de consultas al storage
Comparar una imagen contra 80,000 objetos implica un alto costo de I/O. Implementar una **capa de caché** (e.g. Redis) con los descriptores preprocesados reduce drásticamente el número de lecturas al storage.

### Particionamiento del dataset
Fraccionar las imágenes por región, sede u origen de la solicitud evita comparar contra la totalidad del dataset en cada consulta, mejorando el tiempo de respuesta y reduciendo costos de cómputo.
