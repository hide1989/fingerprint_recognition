## Analisis del uso de detección de huellas usando ORB

### Motivos encontrados con puntajes bajos

1. Aunque el algoritmo de detección y descripción ORB soporta rotaciones, uno de los motivos de baja puntuación podria deberse a la rotación, si esta es muy elevada **> 45° - 60°**. El algoritmo trabaja bien hasta rotaciones de **<=30°**.
2. Exceso de brillo u obscuridad en la imagen resultante despues de pasarla a una escala de grises. Esto puede afectar el reconocimiento adecuado de los puntos de interes (minucias)
3. Ruido adicional en la imagen como sudor o suciedad en en la huella al momento de extraer la imagen puede generar cambios en las crestas o bifurcaciones que realmente no existen.
4. Este algoritmo es de proposito general, no fue diseñado enfocado en la detección de huellas, por lo que los supuestros geometricos que maneja tienden a fallar mas que algoritmos patentados que estan orientados a esta validación como lo es el **ISO/IEC 19794**
5. Falta de filtros previos a la identificación de los key points; Si se descartan imagenes que no tengan los minimos key points necesarios para realizar una correcta identificación. Se puede aumentar bastante la presesión al descartar imagenes de poca calidad.

### Escalabilidad de 1 a 1000

Hay varias formas de escarlar un sistema, pero este depende de varios factores que intervienen en las decisiones tecnicas para un correcto escalamiento sin entrar en sobre costes o sobrearquitectar la aplicación. Vamos a trabajar sobre el supuesto de que requerimos subir la cantidad de usuarios que tienen registrada su huella.

La aplicación requiere aumentar las imagenes de reconocimiento de 80 a 80,000. esto conlleva a tener en cuenta un correcto almancenamiento. Para este caso más que solo datos requerimos almacenar objetos por lo que servicios de storage en la nube orientado a estos propositos como lo son Amazon S3 de AWS o blob storage de Azure tienden a ser una opción bastante atractiva, Sobretodo si son de acceso frecuente.
Obtener una imagen para comparar contra 80 mil objetos es algo que consume bastantes recursos y aumenta los costos de consulta al storage. Por lo que aplicar una cache favorece la reducción en el numero de pegadas que se realiza al storage para recuperar los datos.

Reducir el numero de imagenes que se consultan franccionandolas por region, sede, origen de la request etc. favorecen el rendimiento al evitar consultar y comparar contra todas las imagenes disponibles.

