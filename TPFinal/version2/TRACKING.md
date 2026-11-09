El programa principal usa `video_recortado.mp4` y conserva YOLOv8s, con
inferencia a 640 px y recortes sobre el frame original. Para este partido,
`UNIFORMES_AZUL_BLANCO = True` usa reglas HSV calibradas sobre recortes reales:
A azul oscuro, B blanco y posibles arbitros gris. No identifica al arbitro
por ser el tercer grupo de K-means; una region ambigua queda sin equipo.
Estas reglas dependen de los uniformes y la iluminacion de este video.
El tono de los shorts debe ser compatible con la camiseta: una parte clara
con pantalon negro o camiseta azul con shorts blancos queda ambigua. Esto
evita usar brazos/fondo claros como evidencia suficiente del equipo blanco.
Por defecto no se carga SigLIP: la camiseta tambien usa un descriptor
normalizado de color y textura. Si estan los pesos locales, OSNet x0.25
(preentrenado en MSMT17) agrega un vector de cuerpo completo cada diez frames
por ID de ByteTrack, en lotes. Su comparacion tambien interviene en la
continuidad para ayudar frente a intercambios de personas del mismo equipo.
`--sin-reid` desactiva OSNet para comparar con color/textura solamente.
`--siglip` habilita la alternativa anterior de apariencia profunda cada
15 frames por ID de ByteTrack, o al aparecer uno nuevo; puede mejorar la
comparacion entre vistas, pero es mucho mas lento en CPU. En el modo
calibrado se omiten el aprendizaje y UMAP por completo.
Con `UNIFORMES_AZUL_BLANCO = False` se habilita el agrupamiento automatico
anterior; su tercer grupo no garantiza que sean arbitros.
Cuerpo, shorts y cabeza usan descriptores baratos de color y textura en
cada frame para asociar personas. La cabeza aporta apariencia
general (pelo, rostro visible); no utiliza un modelo especializado de
reconocimiento facial. Se omiten regiones pequenas o borrosas.

`SEGUNDO_INICIO = 23` inicia el aprendizaje y el procesamiento en 00:23.
`SEGUNDOS_PROCESAR = 12` es la duracion maxima de salida a partir de ese
instante (hasta 00:35 o el final del archivo). La consola informa el tiempo
del video original ademas del progreso del segmento.

Las detecciones sin ID confirmado se muestran en gris con `ID --`;
solo los confirmados usan IDs 1 a 12 y color de equipo. Un recuadro gris
no significa que se haya recuperado la identidad. Las cajas extrapoladas se
rotulan y cuentan por separado de las detecciones. La velocidad se estima con las ultimas
tres observaciones para adaptarse a aceleraciones y cambios de direccion.

La confirmacion inicial sigue usando diez frames y ocho votos. Para IDs ya
confirmados, la continuidad geometrica de ByteTrack y sus aliases recientes
mantienen la identidad durante variaciones breves de apariencia. Una camiseta
ocluida actualiza la posicion sin reemplazar la plantilla visual. Si cambia
el color con histograma incompatible se rechaza inmediatamente; con histograma
similar se toleran dos lecturas contradictorias, no un cambio persistente.
Las detecciones geometricamente coherentes de un ID conocido no se descartan
por el piso o el marcador de TV. Estos filtros siguen aplicandose a candidatos
nuevos para evitar que el personal del borde ocupe cupos.

Cuando falta una deteccion, se puede dibujar una prediccion durante hasta cinco
frames, rotulada `Predicho`; no cuenta como deteccion confirmada ni genera ID.
Ante un primer plano (persona muy grande en imagen), se reservan los IDs y
se invalidan las posiciones. Se muestran recuadros grises con identidad en
pausa. Al volver a la toma abierta, la galeria puede recuperar un ID con tres
observaciones compatibles; una identidad nueva sigue requiriendo diez.
La recuperacion ambigua entre companeros queda sin ID, especialmente despues
de un cambio de plano sin dorsales visibles. No se fuerza un ID por descarte.

Antes de asociar posiciones, se estima paneo/zoom con ORB y RANSAC sobre
gradas y publicidad a 480 px. Se transforman las posiciones historicas al
frame actual. Si faltan correspondencias fiables, no se aplica la matriz.
El color fuerte contrario con histograma incompatible veta la asociacion;
un cambio persistente tambien la veta aunque ByteTrack conserve su ID.
Las lecturas debiles no sustituyen la plantilla de un ID confirmado.

Para activar lectura de dorsales, instalar en el mismo entorno de main.py:

```powershell
python -m pip install easyocr
python -m unittest test_seguimiento.py test_apariencia.py test_reid_ligero.py
python main.py
```

La ejecucion por defecto genera `diagnostico_tracking.mp4` (12 segundos).
Para una prueba sin OCR: `python main.py --segundos 12 --sin-ocr`.
Para comparar con apariencia profunda: `python main.py --siglip --sin-ocr`.
Para comparar sin OSNet: `python main.py --sin-reid --sin-ocr`.
Para procesar el minuto: `python main.py --segundos 60 --salida resultado_equipos_mejorado.mp4`.
Se guardan capturas cada tres segundos, el ultimo frame y `metricas.csv` en
una carpeta con el nombre de la salida, sin extension. Las metricas separan
las detecciones de YOLO, las aptas para tracking, las confirmadas y las
recuperaciones de identidades perdidas, ademas de si se pudo compensar la
camara. `trayectorias.csv` guarda por frame el ID interno, ID de ByteTrack,
equipo bloqueado, equipo observado y coordenadas del recuadro. El contador
`Recuperados` mide fusiones por galeria; las asociaciones directas pueden
conservar un ID aunque ese contador siga en cero.
Las columnas `ms_detector`, `ms_camara`, `ms_apariencia_ocr` y `ms_asociacion`
permiten localizar el costo de CPU por etapa. OpenCV usa un hilo para no
competir con el pool de PyTorch; no se limita el paralelismo de PyTorch.

OSNet se distribuye en `vendor/osnet.py` con licencia MIT y pesos oficiales
de 9.34 MB en `modelos/osnet_x025_msmt17.pth`. Ver fuentes y hashes en
`vendor/README.md`. El cargador usa `weights_only=True`, excluye la cabeza
de clasificacion y comprueba que las capas de inferencia sean compatibles.
No requiere instalar torchreid ni realiza descargas durante el seguimiento.
Se evita actualizar OSNet y la galeria con recuadros solapados (IoU > 0.20).

`evaluar_reid.py` compara tres segundos tras el corte (01:11-01:14) usando
vistas de trayectorias previas a 01:03. Es una prueba controlada de recuperacion,
no una evaluacion de exactitud: los IDs de la galeria proceden de tracking
anterior y no son anotaciones manuales verdaderas. Los conteos recuperados no
garantizan que todas las identidades sean correctas. No se usan frames futuros.

Una trayectoria nueva con al menos tres observaciones compatibles puede recuperar un ID perdido
por comparacion de apariencia con una galeria de hasta cuarenta vistas.
Se exige el mismo equipo, coincidencia en al menos dos regiones y diferencia
suficiente respecto al segundo candidato; dorsales confirmados contradictorios
vetan la recuperacion. Se conservan los cupos y no se recupera un ID ya
asignado a otra deteccion del frame. Esta recuperacion admite cambios de ID
de ByteTrack y desplazamientos fuera del radio de asociacion espacial.

`FILTRAR_PISO_CANCHA = True` estima el parquet en tomas abiertas y filtra
personas cuyos pies no estan dentro de la superficie estimada, con un margen
interior del 0.6% del ancho para excluir personal pegado al borde. Puede
omitir jugadores muy cerca de la linea lateral. Si no encuentra
una region suficientemente grande, omite ese filtro. Es una heuristica de
color para este video: un piso distinto o un cambio de iluminacion puede
requerir desactivarla. No equivale a una calibracion exacta de la cancha.
En tomas abiertas tambien se revisa un parche debajo de los pies: debe
parecer parquet o pintura roja, para rechazar la banda azul y la publicidad
que quedaron dentro de la envolvente. Un posible arbitro gris fiable puede
estar en la banda lateral sin superar ese segundo filtro.

En el modo de agrupamiento automatico, el aprendizaje se guarda en
`uniformes_cache.npz` y se reutiliza si coinciden
el video, el segundo inicial y los parametros relevantes. Cambiar la duracion
de salida o desactivar OCR no obliga a aprender de nuevo. UMAP se importa
solo cuando se necesita aprender, para acelerar el arranque con cache.

OCR queda desactivado por defecto para conservar velocidad en CPU.
EasyOCR puede descargar sus pesos en el primer inicio. Si no esta disponible,
el programa avisa y sigue con las otras pistas; `USAR_OCR_DORSALES = False`
permite desactivarlo explicitamente. La lectura se hace cada 30 frames,
con hasta dos recortes por turno, priorizando completar lecturas pendientes.
Solo acepta uno o dos digitos con confianza >= 0.85, de tamano suficiente,
y exige tres lecturas iguales en frames distintos dentro de cinco segundos.
Repetir un resultado almacenado no cuenta como una nueva lectura.
Se conserva el texto para distinguir, por ejemplo, 0 y 00. El dorsal aparece
en la etiqueta junto al ID, que sigue siendo un numero interno entre 1 y 12.
No leer un dorsal no impide seguir al jugador; una lectura contradictoria
individual tampoco veta por si sola una asociacion.

Para activar explicitamente esta comprobacion:

```powershell
python main.py --segundos 12 --verificar-dorsales --salida tracking_verificado.mp4
```

No combinar `--verificar-dorsales` con `--sin-ocr`. El presupuesto es de dos
recortes cada 30 frames (dos por segundo en este video), pero OCR agrega
tiempo de CPU. Un consenso OCR es evidencia automatica, no validacion humana.
El dorsal fijado se conserva por identidad y equipo; lecturas aisladas no
lo reemplazan. Si no hay lectura suficiente, continua la asociacion visual.

Las referencias OSNet iniciales se conservan: una asociacion dudosa no debe
reemplazarlas gradualmente por la apariencia de otro jugador. Nuevas vistas
requieren similitud y margen frente a rivales, o un dorsal coincidente.
Cada ID de ByteTrack tiene un unico propietario confirmado; las cajas muy
solapadas del mismo equipo se suprimen antes de extraer apariencia.

La consola dice `IDs asociados`: ese conteo no mide exactitud de identidad.
`Det` en las etiquetas es confianza del detector. `Dorsal ... OCR` indica
consenso de lecturas, sin garantizar que todas sean correctas.
La auditoria manual y sus limites estan en `VALIDACION_IDS.md`.
`comparar_referencias.py --trayectorias carpeta/trayectorias.csv` contrasta
las muestras anotadas por equipo y dorsal contra una nueva salida.

Se mantiene la confirmacion de equipo con 8 votos en 10 frames consecutivos,
los cupos de cinco jugadores por equipo y dos posibles arbitros y la memoria
de asociacion espacial de tres segundos. Luego se permite recuperar un ID
con continuidad de ByteTrack o dorsal confirmado y apariencia muy similar,
o mediante la galeria tras confirmar una nueva trayectoria de diez frames.
Los tracks confirmados se asocian antes que los candidatos y un voto aislado
de equipo contradictorio no rompe una trayectoria continua de ByteTrack.
La estimacion de arbitros por color no es un detector entrenado de arbitros. Los cupos
suponen los mismos doce participantes durante el segmento: no implementan
sustituciones. Cambios de plano, oclusiones largas y uniformes muy similares
todavia pueden causar perdida o intercambio de identidad. Los embeddings
generales y el OCR no garantizan identificar cada jugador.

La consola informa frames/s y tiempo restante aproximado cada 100 frames.
El modo rapido mantiene todos los frames de salida, pero no mide de nuevo
SigLIP en cada frame. Puede perder detalle de personas pequenas con YOLO
a 640 px. Los umbrales y pesos son iniciales: requieren validacion sobre el
video. La reduccion de llamadas no equivale a una aceleracion medida.
